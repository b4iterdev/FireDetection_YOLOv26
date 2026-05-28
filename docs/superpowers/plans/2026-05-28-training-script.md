# Training Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a `scripts/train.py` script to fine-tune YOLOv26 on the D-Fire dataset and auto-deploy the results.

**Architecture:** A CLI tool that loads project defaults, runs Ultralytics training, and atomically deploys the best model to `models/fire_yolov26.pt`.

**Tech Stack:** Python, Ultralytics YOLO, PyYAML, shutil, argparse.

---

### Task 1: Training Script Skeleton and Argument Parsing

**Files:**
- Create: `scripts/train.py`
- Modify: `configs/default.yaml` (verify content)

- [ ] **Step 1: Implement basic CLI structure and config loading**

```python
import argparse
import os
import shutil
from ultralytics import YOLO
from fire_detection_alarm.app.config import load_config

def parse_args(cfg):
    parser = argparse.ArgumentParser(description="Train YOLOv26 on D-Fire dataset")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model to train")
    parser.add_argument("--data", default="dataset/D-Fire/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=cfg.get("model", {}).get("image_size", 640), help="Image size")
    parser.add_argument("--device", default=cfg.get("model", {}).get("device", "auto"), help="Device (cpu, 0, etc.)")
    parser.add_argument("--project", default="runs/train", help="Project directory")
    parser.add_argument("--name", default="fire_detection", help="Experiment name")
    return parser.parse_args()

def main():
    cfg = load_config()
    args = parse_args(cfg)
    print(f"Starting training with base model: {args.model}")
    print(f"Dataset: {args.data}")
    # Placeholder for Task 2
    
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run: `python scripts/train.py --help`
Expected: Help message displayed with correct defaults.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: add training script skeleton and arg parsing"
```

---

### Task 2: Implement Training and Deployment Logic

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Add training and deployment code**

```python
// ... existing code ...
def main():
    cfg = load_config()
    args = parse_args(cfg)
    
    # 1. Initialize model
    model = YOLO(args.model)
    
    # 2. Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True
    )
    
    # 3. Deploy best model
    best_model_path = os.path.join(args.project, args.name, "weights", "best.pt")
    target_path = "models/fire_yolov26.pt"
    
    if os.path.exists(best_model_path):
        os.makedirs("models", exist_ok=True)
        shutil.copy2(best_model_path, target_path)
        print(f"\nSUCCESS: Training complete.")
        print(f"Deployed best model to: {target_path}")
    else:
        print(f"\nERROR: Training finished but {best_model_path} not found.")

if __name__ == "__main__":
// ... existing code ...
```

- [ ] **Step 2: Dry run with 1 epoch**

Run: `python scripts/train.py --epochs 1 --batch 1`
Expected: Training starts, completes 1 epoch, and attempts to deploy.
*Note: Since the dataset is large, this is just to verify the plumbing.*

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: implement training and auto-deployment logic"
```
