import os
import yaml
from fire_detection_alarm.app.config import load_config


def test_load_default_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "default.yaml"
    data = {
        "model": {"path": "models/fire.pt"},
        "inference": {"confidence_threshold": 0.5},
    }
    with open(config_file, "w") as f:
        yaml.dump(data, f)

    cfg = load_config(str(config_file))
    assert cfg["model"]["path"] == "models/fire.pt"
    assert cfg["inference"]["confidence_threshold"] == 0.5
