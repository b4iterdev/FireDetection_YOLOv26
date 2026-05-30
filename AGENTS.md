# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-29T07:12:00Z
**Commit:** 958d756
**Branch:** main

## OVERVIEW
An end-to-end modular fire/smoke detection pipeline utilizing Ultralytics YOLOv26. It handles stream/file decoding, model inference, bounding box normalization, temporal status aggregation/filtering, and structured alerting.

## STRUCTURE
```
{root}/
├── configs/                  # Pipeline and threshold parameters
│   └── default.yaml          # Config values (model path, device, confidence, etc.)
├── scripts/                  # Running/training entry points
│   ├── demo_offline.py       # Main system orchestrator / entrypoint
│   └── train.py              # Model training orchestration utilities
├── fire_detection_alarm/     # Core package
│   ├── app/                  # Application configuration
│   ├── models/               # YOLO inference wrapper
│   ├── detection/            # Schema definition, normalization, and rendering
│   ├── filtering/            # Temporal logic, decision, and cooldown gating
│   └── logging/              # Structured logging for system events
└── tests/                    # Pipeline validation tests (uses pytest & YOLO mocks)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Pipeline Execution | `scripts/demo_offline.py` | Controls inputs -> inference -> filtering -> output loop |
| YOLO Inference Wrapper | `fire_detection_alarm/models/yolo_engine.py` | Handles `predict()` via `ultralytics.YOLO` |
| Bounding Box Mapping | `fire_detection_alarm/detection/normalizer.py` | Maps raw bounding boxes into unified `Detection` schemas |
| Spatiotemporal Filtering | `fire_detection_alarm/filtering/` | Spatial logic (bbox area checks), temporal logic (persistence) |
| Configuration | `fire_detection_alarm/app/config.py` | Reads and maps configs from yaml to app settings |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Detection` | Class | `fire_detection_alarm/detection/schema.py` | Canonical model representation of a frame prediction |
| `YOLOEngine` | Class | `fire_detection_alarm/models/yolo_engine.py` | Performs frame inference, configurable via device/imgsz |
| `normalize_yolo_output` | Function | `fire_detection_alarm/detection/normalizer.py` | Parses raw tensor models to structured `Detection` format |
| `DetectionFilter` | Class | `fire_detection_alarm/filtering/detection_filter.py` | Filters predictions by labels, confidence, and box area ratio |
| `TemporalFilter` | Class | `fire_detection_alarm/filtering/temporal_filter.py` | Aggregates across multiple frames to confirm persistent alarms |
| `BaseSource` | Class | `fire_detection_alarm/inputs/base.py` | Base abstract source contract (`read()`, `release()`) |
| `render_detections` | Function | `fire_detection_alarm/detection/renderer.py` | Annotates frames with detected bounding boxes and labels |

## CONVENTIONS
- **Canonical Data Schema**: Always pass lists of `Detection` instances rather than raw dictionary/list objects between pipeline stages.
- **Model Classes**: The pipeline strictly expects indices `0` (smoke) and `1` (fire) mapped from datasets.
- **Source Interface**: New stream components (e.g., RTSP, webcam) must inherit from `BaseSource` and implement `read()` / `release()`.

## ANTI-PATTERNS (THIS PROJECT)
- **Hardcoding Configuration**: Never hardcode image size (imgsz), device (`cpu`, `cuda`), or paths. Always parse via `app.config`.
- **Suppressing Type Checks**: Do not use `as any` or disable types; maintain explicit python type annotation correctness.
- **Tracking Model Weights**: Never commit YOLO models (e.g. `.pt` weights under `models/`) directly into the repository.
- **Bypassing Normalization**: Never feed raw outputs from models straight to filters without passing through the normalizer first.

## UNIQUE STYLES
- **Interchangeable Naming**: `YOLOv26` and `YOLO26` are used interchangeably to refer to Ultralytics-based models.
- **Zero-GPU Testing**: Tests use `unittest.mock` to mock `YOLO` engine predictions to allow complete pipeline unit testing in standard CI environments without GPU dependencies.

## COMMANDS
```bash
# Setup dependencies (inside project venv)
venv/bin/pip install ultralytics torch torchvision opencv-python pyyaml pytest

# Run demo with external weights (not tracked in Git)
PYTHONPATH=. venv/bin/python scripts/demo_offline.py --input path/to/video.mp4 --model models/fire_yolov26.pt

# Run pytest suite (package imports require PYTHONPATH)
PYTHONPATH=. venv/bin/python -m pytest -q
```

## NOTES
- **Image Size Sync**: `configs/default.yaml` specifies `image_size: 640`, but the offline demo defaults to `640` internally if not explicitly defined. Always check parameter synchronization.
- **Runtime Environment**: Use the repo `venv` and set `PYTHONPATH=.` when running scripts/tests from repo root so `fire_detection_alarm` imports resolve correctly.
