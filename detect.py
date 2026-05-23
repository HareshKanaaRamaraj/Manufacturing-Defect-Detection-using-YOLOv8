"""
detect.py — Real-time Manufacturing Defect Detection on Raspberry Pi
=====================================================================
Model  : YOLOv8n exported to ONNX (trained on NEU Surface Defect Dataset)
Device : Raspberry Pi 4 (4GB RAM recommended)
Camera : USB webcam (index 0) or Pi Camera Module v2 (see --source flag)
Output : Live annotated feed + detections.csv log

Usage:
    python detect.py                        # default settings
    python detect.py --source 1             # camera index 1
    python detect.py --model custom.onnx    # different model
    python detect.py --conf 0.4             # stricter confidence
    python detect.py --no-display           # headless / no monitor

Requirements:
    pip install onnxruntime opencv-python numpy
"""

import cv2
import csv
import time
import argparse
import numpy as np
import onnxruntime as ort

from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  Configuration — edit here or pass as CLI arguments
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "/home/pi/defect_detection/best.onnx"
DEFAULT_SOURCE  = 0            # camera index; or path to video file
DEFAULT_CONF    = 0.25         # minimum detection confidence
DEFAULT_IOU     = 0.45         # NMS IoU threshold
DEFAULT_LOGFILE = "/home/pi/defect_detection/detections.csv"
IMG_SIZE        = 640          # must match training imgsz

# 6 NEU-DET defect classes (order = class id)
CLASSES = [
    "crazing",         # 0 — network of fine surface cracks
    "inclusion",       # 1 — foreign particles embedded in steel
    "patches",         # 2 — irregular discoloured regions
    "pitted_surface",  # 3 — small pits / holes
    "rolled-in_scale", # 4 — scale pressed in during rolling
    "scratches",       # 5 — linear grooves or scratches
]

# BGR colours for bounding boxes (one per class)
COLORS_BGR = [
    (50,  50,  220),   # crazing         — red
    (50,  180, 50),    # inclusion       — green
    (220, 50,  50),    # patches         — blue
    (0,   180, 220),   # pitted_surface  — yellow
    (220, 180, 0),     # rolled-in_scale — cyan
    (220, 0,   180),   # scratches       — purple
]


# ─────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────
def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model and return inference session."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "  Transfer best.onnx from Kaggle:\n"
            "    scp best.onnx pi@<pi-ip>:/home/pi/defect_detection/"
        )

    print(f"Loading model : {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"  Input  : {inp.name}  shape={inp.shape}  dtype={inp.type}")
    print(f"  Output : {out.name}  shape={out.shape}  dtype={out.type}")
    print(f"  ONNX Runtime {ort.__version__}")
    return session


# ─────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray):
    """
    Prepare BGR frame for YOLOv8 ONNX inference.

    Uses letterbox resize to preserve aspect ratio.

    Returns:
        blob    : float32 tensor  [1, 3, IMG_SIZE, IMG_SIZE]
        scale_x : width  scale factor (map model coords → original)
        scale_y : height scale factor
    """
    orig_h, orig_w = frame.shape[:2]

    # Letterbox
    scale  = IMG_SIZE / max(orig_h, orig_w)
    new_w  = int(orig_w * scale)
    new_h  = int(orig_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # BGR → RGB, normalise [0,1], HWC → CHW, add batch dim
    rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    # Scale factors to map back to original frame size
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    return blob, scale_x, scale_y


# ─────────────────────────────────────────────────────────────
#  Non-Maximum Suppression
# ─────────────────────────────────────────────────────────────
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    """Greedy NMS. Returns list of kept indices."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]

    return keep


# ─────────────────────────────────────────────────────────────
#  Postprocessing
# ─────────────────────────────────────────────────────────────
def postprocess(outputs, scale_x, scale_y, conf_thresh, iou_thresh):
    """
    Parse raw YOLOv8 ONNX output.

    YOLOv8 ONNX output shape: [1, (4 + nc), 8400]
      For nc=6:  [1, 10, 8400]
    """
    raw = outputs[0][0]      # [10, 8400]
    raw = raw.T              # [8400, 10]

    boxes_xywh  = raw[:, :4]
    class_probs = raw[:, 4:]

    class_ids   = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs,    axis=1)

    # Confidence filter
    mask        = confidences >= conf_thresh
    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_xywh) == 0:
        return np.array([]), np.array([]), np.array([])

    # xywh (model space) → xyxy (original frame space)
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) * scale_x
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) * scale_y
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) * scale_x
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) * scale_y
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes_xyxy, confidences, iou_thresh)
    return boxes_xyxy[keep], confidences[keep], class_ids[keep]


# ─────────────────────────────────────────────────────────────
#  Drawing Utilities
# ─────────────────────────────────────────────────────────────
def draw_detections(frame, boxes, scores, class_ids):
    """Draw bounding boxes and labels on frame in-place."""
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cls_id   = int(cls_id)
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        color    = COLORS_BGR[cls_id % len(COLORS_BGR)]
        label    = f"{cls_name}  {score:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label pill
        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )
        ly1 = max(y1 - th - bl - 4, 0)
        cv2.rectangle(frame, (x1, ly1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - bl - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 255), 1, cv2.LINE_AA
        )
    return frame


def draw_hud(frame, n_dets, fps, avg_fps):
    """Draw status bar + FPS overlay."""
    h, w = frame.shape[:2]

    # Top status bar
    if n_dets > 0:
        bar_color   = (0, 0, 200)
        status_text = f"  DEFECT DETECTED  ({n_dets})"
    else:
        bar_color   = (0, 150, 0)
        status_text = "  OK — No defect"

    cv2.rectangle(frame, (0, 0), (w, 44), (30, 30, 30), -1)
    cv2.putText(
        frame, status_text, (6, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        bar_color, 2, cv2.LINE_AA
    )

    # FPS counter bottom-right
    fps_text = f"FPS: {fps:.1f}  avg: {avg_fps:.1f}"
    cv2.putText(
        frame, fps_text,
        (w - 210, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (180, 180, 180), 1, cv2.LINE_AA
    )
    return frame


# ─────────────────────────────────────────────────────────────
#  Detection Logger
# ─────────────────────────────────────────────────────────────
class DetectionLogger:
    """Appends every detection to a CSV file with timestamp."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "class_id", "class_name",
                "confidence", "x1", "y1", "x2", "y2"
            ])
        print(f"Logging detections → {log_path}")

    def log(self, boxes, scores, class_ids):
        if len(boxes) == 0:
            return
        ts = datetime.now().isoformat(timespec="milliseconds")
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            for box, score, cls_id in zip(boxes, scores, class_ids):
                cls_id   = int(cls_id)
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                w.writerow([
                    ts, cls_id, cls_name,
                    f"{score:.4f}",
                    int(box[0]), int(box[1]),
                    int(box[2]), int(box[3]),
                ])


# ─────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────
def run(args):
    session    = load_model(args.model)
    input_name = session.get_inputs()[0].name

    # Open camera / video source
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open source: {args.source}\n"
            "  Try --source 1 if index 0 fails\n"
            "  Check available cameras: ls /dev/video*"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nCamera         : {actual_w}x{actual_h}")
    print(f"Conf threshold : {args.conf}")
    print(f"IoU  threshold : {args.iou}")
    print(f"Display        : {'OFF (headless)' if args.no_display else 'ON'}")
    print("\nRunning — press Q to quit\n")

    logger      = DetectionLogger(args.log)
    fps_history = []
    frame_count = 0
    total_dets  = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed — exiting")
                break

            t0 = time.perf_counter()

            # ── Inference ──────────────────────────────────────────
            blob, sx, sy = preprocess(frame)
            outputs      = session.run(None, {input_name: blob})
            boxes, scores, class_ids = postprocess(
                outputs, sx, sy, args.conf, args.iou
            )

            elapsed = time.perf_counter() - t0
            fps     = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_history.append(fps)
            avg_fps = float(np.mean(fps_history[-30:]))

            # ── Log ────────────────────────────────────────────────
            logger.log(boxes, scores, class_ids)
            frame_count += 1
            total_dets  += len(boxes)

            # ── Terminal output every 30 frames ────────────────────
            if frame_count % 30 == 0:
                n       = len(boxes)
                det_str = ""
                if n > 0:
                    names   = [CLASSES[int(c)] for c in class_ids]
                    det_str = "  -> " + ", ".join(
                        f"{nm}({sc:.2f})"
                        for nm, sc in zip(names, scores)
                    )
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}]"
                    f"  frame {frame_count:>5}"
                    f"  {fps:.1f} FPS"
                    f"  {n} detection(s){det_str}"
                )

            # ── Display ────────────────────────────────────────────
            if not args.no_display:
                annotated = draw_detections(frame.copy(), boxes, scores, class_ids)
                annotated = draw_hud(annotated, len(boxes), fps, avg_fps)
                cv2.imshow("NEU Defect Detection  |  Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nQ pressed — stopping")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted — stopping")

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        print("\n" + "=" * 52)
        print("  Session Summary")
        print("=" * 52)
        print(f"  Frames processed : {frame_count}")
        print(f"  Total detections : {total_dets}")
        if fps_history:
            print(f"  Avg FPS          : {np.mean(fps_history):.2f}")
            print(f"  Min / Max FPS    : {min(fps_history):.2f} / {max(fps_history):.2f}")
        print(f"  Log saved        : {args.log}")
        print("=" * 52)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time YOLOv8 defect detection — Raspberry Pi"
    )
    p.add_argument("--model",      default=DEFAULT_MODEL,
                   help=f"ONNX model path (default: {DEFAULT_MODEL})")
    p.add_argument("--source",     default=DEFAULT_SOURCE,
                   help="Camera index or video path (default: 0)")
    p.add_argument("--conf",       type=float, default=DEFAULT_CONF,
                   help=f"Confidence threshold (default: {DEFAULT_CONF})")
    p.add_argument("--iou",        type=float, default=DEFAULT_IOU,
                   help=f"NMS IoU threshold (default: {DEFAULT_IOU})")
    p.add_argument("--log",        default=DEFAULT_LOGFILE,
                   help=f"CSV log path (default: {DEFAULT_LOGFILE})")
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode — disable cv2.imshow")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
