# 🏭 AI-Powered Manufacturing Defect Detection + Edge Deployment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=yolo)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-gray?logo=onnx)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-4B-red?logo=raspberrypi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**Real-time steel surface defect detection trained on Kaggle GPU and deployed on Raspberry Pi 4 for edge inference.**

[Overview](#-overview) • [Demo](#-demo) • [Dataset](#-dataset) • [Architecture](#-architecture) • [Setup](#-setup) • [Training](#-training-on-kaggle) • [Deployment](#-edge-deployment--raspberry-pi) • [Results](#-results)

</div>

---

## 📌 Overview

This project implements an end-to-end **Industry 4.0** pipeline for automated visual quality inspection on manufacturing conveyor lines. It detects 6 types of steel surface defects in real time using a YOLOv8 object detection model, trained on Kaggle's free GPU and deployed on a Raspberry Pi 4 for low-cost, offline edge inference.

### Key Highlights

- **6 defect classes** detected from 200×200 grayscale steel surface images
- **YOLOv8n** (nano) — fast and lightweight, optimised for edge hardware
- **ONNX export** — runs on Raspberry Pi 4 at 3–6 FPS with no cloud dependency
- **Real-time alert logging** — every detection timestamped to a CSV file
- **Auto-start on boot** via systemd — production-ready deployment

---

## 🎯 Demo

```
Camera Frame → Preprocess → ONNX Inference → NMS → Draw Boxes → Display
                                                          │
                                                    Log to CSV
                                              (timestamp, class, confidence)
```

Sample output on Raspberry Pi:

```
[2025-05-21 10:32:11] DEFECT DETECTED — inclusion  (conf: 0.87)
[2025-05-21 10:32:11] DEFECT DETECTED — scratches  (conf: 0.74)
[2025-05-21 10:32:14] OK — No defect
Average FPS: 4.3
```

---

## 📂 Dataset

**NEU Surface Defect Database (NEU-DET)**

| Property | Value |
|---|---|
| Source | Northeastern University, China |
| Total images | 2,100 (1,800 train + 300 val) |
| Image size | 200 × 200 px grayscale |
| Annotation format | Pascal-VOC XML |
| Classes | 6 defect types |
| Kaggle link | [neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |

### Defect Classes

| ID | Class | Description |
|---|---|---|
| 0 | `crazing` | Network of fine cracks on surface |
| 1 | `inclusion` | Foreign material embedded in steel |
| 2 | `patches` | Irregular surface discolouration |
| 3 | `pitted_surface` | Small pits/holes on surface |
| 4 | `rolled-in_scale` | Scale pressed into surface during rolling |
| 5 | `scratches` | Linear surface scratches |

### Dataset Folder Structure

```
NEU-DET/
├── train/
│   ├── images/
│   │   ├── crazing/          ← 300 images per class
│   │   ├── inclusion/
│   │   ├── patches/
│   │   ├── pitted_surface/
│   │   ├── rolled-in_scale/
│   │   └── scratches/
│   └── annotations/          ← FLAT folder, all 1800 XMLs together
│       ├── crazing_1.xml
│       ├── inclusion_1.xml
│       └── ...
└── validation/
    ├── images/               ← same class subfolder structure
    └── annotations/          ← FLAT folder, 300 XMLs together
```

> ⚠️ **Important:** Annotation XMLs are in a **flat folder** (no class subfolders). The conversion script handles this correctly.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING  (Kaggle)                       │
│                                                             │
│  NEU-DET Dataset                                            │
│       │                                                     │
│       ▼                                                     │
│  XML → YOLO Conversion  (Pascal-VOC → normalised xc,yc,w,h)│
│       │                                                     │
│       ▼                                                     │
│  YOLOv8n  (pretrained COCO → fine-tuned on NEU-DET)        │
│       │                                                     │
│       ▼                                                     │
│  best.pt  →  Export  →  best.onnx                          │
└─────────────────────────────────────────────────────────────┘
                              │
                         scp / USB
                              │
┌─────────────────────────────────────────────────────────────┐
│                  EDGE INFERENCE  (Raspberry Pi 4)            │
│                                                             │
│  USB / Pi Camera                                            │
│       │                                                     │
│       ▼                                                     │
│  OpenCV frame capture                                       │
│       │                                                     │
│       ▼                                                     │
│  Preprocess: resize 640×640 → normalise → CHW tensor       │
│       │                                                     │
│       ▼                                                     │
│  ONNX Runtime inference  (~200–300 ms/frame)                │
│       │                                                     │
│       ▼                                                     │
│  NMS  →  Draw bounding boxes  →  Display                   │
│       │                                                     │
│       └──►  detections.csv  (timestamp, class, confidence) │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠 Setup

### Prerequisites

| Component | Version |
|---|---|
| Python | 3.10+ |
| ultralytics | 8.3+ |
| PyTorch | 2.6+ |
| OpenCV | 4.x |
| ONNX Runtime | 1.17+ |

### Repository Structure

```
Manufacturing-Defect-Detection-using-YOLOv8/
├── README.md
├── NEU_Defect_YOLOv8.ipynb   ← Training notebook (run on Kaggle)
├── detect.py                               ← Raspberry Pi inference script
├── requirements_training.txt               ← Kaggle/PC training deps
├── requirements_pi.txt                     ← Raspberry Pi runtime deps
└── outputs/                                ← Generated after training
    ├── best.pt
    ├── best.onnx                           ← Deploy this to Pi
    └── results.csv
```

### Install — Training (Kaggle / PC)

```bash
pip install -r requirements_training.txt
```

```
# requirements_training.txt
ultralytics>=8.3.0
opencv-python-headless
albumentations
pyyaml
tqdm
pandas
matplotlib
```

### Install — Raspberry Pi

```bash
pip install -r requirements_pi.txt
```

```
# requirements_pi.txt
onnxruntime
opencv-python
numpy
```

---

## 🚀 Training on Kaggle

### Step 1 — Add Dataset

1. Go to [Kaggle](https://www.kaggle.com) → Create Notebook
2. Click **+ Add Data** → search `NEU Surface Defect` → Add
3. Set **Settings → Accelerator → GPU T4 x2**

### Step 2 — Run the Notebook

Upload `NEU_Defect_YOLOv8_Kaggle_FIXED.ipynb` and run all 19 cells in order.

| Cell | Action |
|---|---|
| 1 | GPU verification |
| 2 | Install ultralytics (latest) |
| 3–4 | Imports & config |
| 5–6 | Dataset stats & raw visualisation |
| 7–8 | XML → YOLO conversion |
| 9 | **Verify bounding boxes visually** — must look correct before training |
| 10–11 | data.yaml + class distribution |
| 12–13 | Load model + **Train** (~35–45 min) |
| 14–15 | Validate + training curves |
| 16–17 | Inference + confusion matrix |
| 18 | **Export** best.onnx + best_float32.tflite |
| 19 | Final summary & download checklist |

### Step 3 — Download Outputs

From the Kaggle output panel, download:
- `outputs/neu_defect_yolov8/weights/best.onnx` ← **use this on Pi**
- `outputs/neu_defect_yolov8/weights/best.pt` ← keep as backup

### Training Configuration

```python
model      = YOLOv8n          # pretrained on COCO
epochs     = 50
imgsz      = 640
batch      = 16
optimizer  = AdamW
lr0        = 0.001
patience   = 15               # early stopping
device     = GPU T4
```

---

## 🍓 Edge Deployment — Raspberry Pi

### Step 1 — Hardware Setup

| Item | Spec |
|---|---|
| Board | Raspberry Pi 4 (4GB RAM recommended) |
| OS | Raspberry Pi OS 64-bit Bookworm |
| Camera | USB webcam or Pi Camera Module v2 |
| Storage | 32GB+ MicroSD |

### Step 2 — Install Dependencies on Pi

```bash
sudo apt update && sudo apt upgrade -y
pip install onnxruntime opencv-python numpy
```

### Step 3 — Transfer Model

```bash
# Run on your PC
scp best.onnx pi@<pi-ip-address>:/home/pi/defect_detection/
scp detect.py  pi@<pi-ip-address>:/home/pi/defect_detection/
```

### Step 4 — Run Inference

```bash
cd /home/pi/defect_detection
python detect.py
```

Press **Q** to quit. Detections are saved to `detections.csv`.

### Step 5 — Auto-start on Boot (Optional)

```bash
sudo nano /etc/systemd/system/defect-detection.service
```

```ini
[Unit]
Description=NEU Defect Detection
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/defect_detection/detect.py
WorkingDirectory=/home/pi/defect_detection
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable defect-detection
sudo systemctl start defect-detection
```

---

## 📊 Results

### Validation Metrics (YOLOv8n, 50 epochs)

| Metric | Score |
|---|---|
| mAP@0.50 | ~0.78–0.85 |
| mAP@0.50:0.95 | ~0.48–0.55 |
| Precision | ~0.80–0.88 |
| Recall | ~0.75–0.83 |

### Edge Performance (Raspberry Pi 4, 4GB)

| Metric | Value |
|---|---|
| Model format | ONNX (YOLOv8n) |
| Inference speed | ~200–300 ms/frame |
| Throughput | 3–6 FPS |
| RAM usage | ~180 MB |
| Model size | ~6 MB |

> 3–6 FPS is sufficient for conveyor belt inspection at 0.5–1 m/s.

---

## 🔧 Troubleshooting

### `UnpicklingError` when loading model on Kaggle

PyTorch 2.6 changed `torch.load` default to `weights_only=True`. Fix:

```python
import torch, functools
_orig = torch.load
@functools.wraps(_orig)
def _patched(f, *args, **kwargs):
    kwargs["weights_only"] = False
    return _orig(f, *args, **kwargs)
torch.load = _patched
```

### Blank images in bbox verification cell

Your dataset has **flat annotation folders** — no class subfolders inside `annotations/`. The fixed notebook handles this correctly. If you see blank images, confirm your `DATASET_ROOT` path is correct by running:

```python
import os
print(os.listdir("/kaggle/input"))
```

### Camera not found on Pi

```bash
# Check camera index
ls /dev/video*
# Try index 1 if 0 fails
cap = cv2.VideoCapture(1)
```

### Low FPS on Pi

- Make sure no other heavy processes are running: `htop`
- Use `INT8` quantised TFLite model instead of ONNX for ~2× speedup
- Reduce input resolution: change `IMG_SIZE = 320` (trades accuracy for speed)

---

## 🗺 Roadmap

- [x] YOLOv8 training pipeline on Kaggle
- [x] Pascal-VOC → YOLO conversion (flat annotation structure)
- [x] ONNX export for edge deployment
- [x] Real-time inference script for Raspberry Pi
- [x] Detection logging to CSV
- [ ] Flask web dashboard — live feed viewable from browser on local network
- [ ] GPIO buzzer/LED alert on defect detection
- [ ] INT8 quantisation for faster Pi inference
- [ ] TensorRT export for NVIDIA Jetson Nano deployment
- [ ] Docker container for easy deployment

---

## 💼 Industry Relevance

This project demonstrates skills directly applicable to:

| Domain | Application |
|---|---|
| **Industry 4.0** | Automated visual quality inspection |
| **Edge AI** | Offline inference on low-cost hardware |
| **Computer Vision** | Object detection, bounding box regression |
| **MLOps** | Training pipeline, model export, deployment |
| **Embedded Systems** | Raspberry Pi, ONNX Runtime, real-time processing |

Relevant roles: **Manufacturing AI Engineer**, **Computer Vision Engineer**, **Edge AI Engineer**, **ML Engineer (Industrial)**

Companies actively working in this space: Bosch, Siemens, Cognex, Landing AI, Instrumental, Sight Machine.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The NEU Surface Defect Dataset is provided by Northeastern University for research purposes. Please cite the original paper if you use it:

> Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. *Applied Surface Science*, 285, 858-864.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection framework
- [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkc/en/) — dataset
- [ONNX Runtime](https://onnxruntime.ai/) — edge inference engine
- [Kaggle](https://www.kaggle.com) — free GPU training environment

---

<div align="center">
Built for Industry 4.0 · Trained on Kaggle · Deployed on Raspberry Pi
</div>
