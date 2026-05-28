# FireDetection_YOLOv26 Agent Orientation

## Core Architecture
This repository implements a modular fire detection pipeline using **Ultralytics YOLO** (specifically referenced as YOLOv26/YOLO26).

**Data Flow:**
`Input Source` (Image/Video) -> `YOLOEngine` (Inference) -> `Normalizer` (Schema) -> `TemporalFilter` (Logic) -> `Renderer` (Visuals)

- **Entrypoint:** `scripts/demo_offline.py` - Runs inference on a file and displays results.
- **Inference Wrapper:** `fire_detection_alarm/models/yolo_engine.py` (thin wrapper around `ultralytics.YOLO`).
- **Data Model:** `fire_detection_alarm/detection/schema.py` defines the `Detection` dataclass used across the pipeline.
- **Configuration:** `configs/default.yaml` stores thresholds, model paths, and device settings.

## Developer Commands

### Environment Setup
No `requirements.txt` or `pyproject.toml` is currently tracked. Use:
```bash
pip install ultralytics torch torchvision opencv-python pyyaml pytest
```

### Running the Demo
```bash
python scripts/demo_offline.py --input <path_to_video_or_image> --model models/fire_yolov26.pt
```
*Note: The model weights `models/fire_yolov26.pt` are NOT tracked in the repo and must be provided externally.*

### Testing
Tests are located in `tests/` and use `pytest`.
```bash
pytest -q                         # Run all tests
pytest tests/test_yolo_engine.py  # Run specific test file
```
*Note: `test_yolo_engine.py` mocks the YOLO model, so tests do not require a GPU or real weights.*

## Repo-Specific Quirks
- **YOLOv26/YOLO26:** The repo uses these terms interchangeably. It refers to the newer Ultralytics end-to-end models.
- **Config Sync:** `configs/default.yaml` defines `image_size: 640`, but the demo currently defaults to 640 internally rather than explicitly passing this config value to the model.
- **Device Handling:** The config uses `device: "auto"`. Ultralytics handles this by checking for CUDA availability.
- **Model Classes:** The system expects classes `[0: smoke, 1: fire]` as defined in `dataset/D-Fire/data.yaml`.

## Key Documentation
- `docs/superpowers/specs/2026-05-27-fire-detection-alarm-system-design.md`: Full system architecture and future roadmap (FalseNet, RTSP, Cloud).
- `docs/superpowers/plans/2026-05-27-fire-detection-milestone-1-2.md`: Current implementation progress and tech stack.
