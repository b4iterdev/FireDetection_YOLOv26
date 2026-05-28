# Training Script Design - 2026-05-28

## Goal
Create a robust training script `scripts/train.py` that fine-tunes a base YOLOv26 model on the D-Fire dataset and automatically deploys the best model to the project's standard model location.

## Architecture
The script is a standalone utility that integrates with the project's configuration system to ensure consistency between training and inference environments.

### Data Flow
1. **Config Loading**: Read `configs/default.yaml` for default image size, device, and paths.
2. **Argparse**: Override defaults with user-provided CLI arguments.
3. **Training**: Use `ultralytics.YOLO` to train on `dataset/D-Fire/data.yaml`.
4. **Post-processing**: Locate the resulting `best.pt` in the `runs/` directory.
5. **Deployment**: Atomic copy of `best.pt` to `models/fire_yolov26.pt`.

## Technical Details

### Command-Line Arguments
- `--model`: Base model name (default: `yolo11n.pt` - used as YOLOv26 equivalent in current ultralytics).
- `--data`: Path to dataset config (default: `dataset/D-Fire/data.yaml`).
- `--epochs`: Number of training epochs (default: 100).
- `--batch`: Batch size (default: 16).
- `--imgsz`: Training image size (default: 640).
- `--device`: Device to run on (default: `auto`).
- `--project`: Output directory for runs (default: `runs/train`).
- `--name`: Experiment name (default: `fire_detection`).

### Implementation Strategy
- Use `shutil.copy2` for deployment to preserve metadata.
- Ensure the `models/` directory exists before deployment.
- Print a clear success message with the path to the deployed model.

## Success Criteria
- Script executes without syntax errors.
- Training starts correctly and uses the specified dataset.
- The `models/fire_yolov26.pt` file is updated after a successful training run.
- CLI arguments correctly override YAML defaults.
