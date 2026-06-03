# FireDetection YOLOv26

FireDetection YOLOv26 is a modular fire and smoke detection pipeline built around an Ultralytics YOLO model. It reads image, video, webcam, or RTSP inputs; normalizes YOLO detections; applies confidence, spatial, behavior, and temporal filters; then renders/logs accepted detections.

## Requirements

- Python 3.10+
- A YOLO fire/smoke model weights file, for example `models/fire_yolov26.pt`
- A working camera, video file, image file, or RTSP stream for inference

Model weights are not tracked in this repository. Keep `.pt` files outside Git history and place your selected weights at the path used by `configs/default.yaml` or pass the model path explicitly when running scripts.

## Setup with venv

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need a specific CUDA-enabled PyTorch build, install the matching `torch` and `torchvision` wheels for your platform before or after installing the rest of the requirements. See the official PyTorch and Ultralytics installation guidance for GPU-specific commands.

## Model weights

The default config expects:

```text
models/fire_yolov26.pt
```

Create the directory and add your weights:

```bash
mkdir -p models
# copy or download your model to models/fire_yolov26.pt
```

You can also train and deploy weights with:

```bash
PYTHONPATH=. .venv/bin/python scripts/train.py --data dataset/D-Fire/data.yaml
```

Training requires a prepared dataset and a compatible PyTorch/Ultralytics installation.

## Run the offline demo

Use an image or video input:

```bash
PYTHONPATH=. .venv/bin/python scripts/demo_offline.py \
  --input path/to/video.mp4 \
  --model models/fire_yolov26.pt
```

The demo opens an OpenCV window with annotated accepted detections. Press `q` to stop playback.

## Run the web interface

Start Flask:

```bash
PYTHONPATH=. .venv/bin/python scripts/web_app.py
```

Open:

```text
http://127.0.0.1:5000/live
```

The live UI supports webcam, uploaded video playback, local video files, and RTSP streams. Uploaded still images are processed through the batch pipeline and written under `outputs/web/results/`.

## Run tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

The tests mock YOLO model inference where needed, so they do not require a GPU or real model weights for the standard test suite.

## Configuration

Default settings live in `configs/default.yaml`:

- `model.path`: default model weights path
- `model.device`: `cpu`, `cuda`, `mps`, or another Ultralytics-supported device value
- `model.image_size`: inference image size
- `inference.confidence_threshold`: YOLO confidence threshold
- `inference.iou_threshold`: YOLO IoU threshold
- `filtering.*`: persistence and bounding-box filtering thresholds
- `behavior_tracking.*`: growth/stability filtering thresholds
- `classes.allowed`: allowed class IDs; this project expects `0` for smoke and `1` for fire

## Troubleshooting

- If imports fail, make sure commands are run from the repo root with `PYTHONPATH=.`.
- If OpenCV fails on Linux with missing GUI/system libraries, install the required OS packages for your distribution or switch to a headless/server environment where display windows are not needed.
- If PyTorch cannot use your GPU, reinstall `torch` and `torchvision` with the correct CPU/CUDA/MPS support for your platform.
- If the app starts but detects nothing, confirm that the model path exists and that the model was trained with class IDs matching the config.
