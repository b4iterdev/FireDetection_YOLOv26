#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fire_detection_alarm.app.config import load_config
except Exception:  # pragma: no cover - fallback when yaml isn't installed
    def load_config(config_path: str = "configs/default.yaml") -> dict[str, dict[str, str | int]]:
        _ = config_path
        return {
            "model": {
                "path": "models/fire_yolov26.pt",
                "device": "auto",
                "image_size": 640,
            }
        }

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - environment fallback
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )

class TrainableYOLO(Protocol):
    def __init__(self, model: str) -> None: ...

    def train(self, **kwargs: object) -> object: ...


class _MissingYOLO:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._error: ImportError = ImportError("ultralytics is required to train the model")

    def train(self, **_kwargs: object) -> object:
        raise self._error


try:
    from ultralytics.models.yolo.model import YOLO as _UltralyticsYOLO
except ImportError:  # pragma: no cover - environment fallback
    _UltralyticsYOLO = _MissingYOLO

YOLO = cast(type[TrainableYOLO], _UltralyticsYOLO)


DEFAULT_DATA_PATH = "dataset/D-Fire/data.yaml"

ModelConfig = dict[str, str | int]
Config = dict[str, ModelConfig]


class TrainArgs(Protocol):
    model: str
    data: str
    epochs: int
    batch: int
    imgsz: int
    device: str
    project: str
    name: str


def get_default_device(cfg: Config) -> str:
    model_cfg = cfg.get("model", {})
    configured_device = str(model_cfg.get("device", "auto"))

    if configured_device != "auto":
        return configured_device

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args(cfg: Config, argv: list[str] | None = None) -> TrainArgs:
    model_cfg = cfg.get("model", {})
    parser = argparse.ArgumentParser(description="Fire detection training entrypoint")
    _ = parser.add_argument("--model", default=model_cfg.get("path", "yolo11n.pt"), help="Pre-trained YOLOv11/26 model (e.g. yolo11n.pt)")
    _ = parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    _ = parser.add_argument("--epochs", type=int, default=100)
    _ = parser.add_argument("--batch", type=int, default=16)
    _ = parser.add_argument("--imgsz", type=int, default=model_cfg.get("image_size", 640))
    _ = parser.add_argument("--device", default=get_default_device(cfg))
    _ = parser.add_argument("--project", default="runs/train")
    _ = parser.add_argument("--name", default="fire_yolo_train")
    return cast(TrainArgs, cast(object, SimpleNamespace(**vars(parser.parse_args(argv)))))


def main():
    cfg = load_config()
    args = parse_args(cfg)

    print(f"Base model: {args.model}")
    print(f"Dataset: {args.data}")

    try:
        model = YOLO(args.model)
        _ = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )

        best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
        target_path = Path("models") / "fire_yolov26.pt"

        if not best_model_path.exists():
            print(f"ERROR: Training finished but {best_model_path} was not found.")
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(best_model_path, target_path)
        print("SUCCESS: Training complete.")
        print(f"Deployed best model to: {target_path}")
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"ERROR: Training or deployment failed: {exc}")


if __name__ == "__main__":
    main()
