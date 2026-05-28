import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TrainScriptTests(unittest.TestCase):
    def test_get_default_device_prefers_mps(self):
        import scripts.train as train

        with patch("scripts.train.torch.backends.mps.is_available", return_value=True), patch(
            "scripts.train.torch.cuda.is_available", return_value=True
        ):
            self.assertEqual(train.get_default_device({}), "mps")

    def test_parse_args_uses_config_defaults(self):
        import scripts.train as train

        cfg = {
            "model": {"path": "models/fire_yolov26.pt", "device": "auto", "image_size": 640},
        }
        with patch("sys.argv", ["train.py", "--epochs", "10", "--batch", "8"]):
            args = train.parse_args(cfg)

        self.assertEqual(args.model, "models/fire_yolov26.pt")
        self.assertEqual(args.data, "dataset/D-Fire/data.yaml")
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch, 8)
        self.assertEqual(args.imgsz, 640)

    def test_main_trains_copies_best_model_and_reports_success(self):
        import scripts.train as train

        with self.subTest("setup"):
            run_dir = Path(self._testMethodName + "_runs")
            best_model = run_dir / "fire_yolo_train" / "weights" / "best.pt"
            best_model.parent.mkdir(parents=True, exist_ok=True)
            _ = best_model.write_text("weights")

        args = SimpleNamespace(
            model="yolo.pt",
            data="dataset/D-Fire/data.yaml",
            epochs=1,
            batch=1,
            imgsz=640,
            device="cpu",
            project=str(run_dir),
            name="fire_yolo_train",
        )

        train_kwargs: dict[str, object] = {}

        def fake_train(**kwargs: object) -> dict[str, object]:
            train_kwargs.update(kwargs)
            return kwargs

        fake_model = SimpleNamespace(train=fake_train)

        with patch("scripts.train.load_config", return_value={}), patch(
            "scripts.train.parse_args", return_value=args
        ), patch("scripts.train.YOLO", return_value=fake_model), patch(
            "scripts.train.shutil.copy2"
        ) as copy2, patch("scripts.train.Path.exists", return_value=True), patch(
            "builtins.print"
        ) as print_mock:
            train.main()

        copy2.assert_called_once_with(best_model, Path("models") / "fire_yolov26.pt")
        self.assertEqual(train_kwargs["epochs"], 1)
        self.assertEqual(train_kwargs["batch"], 1)
        self.assertEqual(train_kwargs["project"], str(run_dir))
        self.assertEqual(train_kwargs["name"], "fire_yolo_train")
        print_mock.assert_any_call("SUCCESS: Training complete.")
