from pathlib import Path


def test_cooldown_mechanism_removed_from_runtime_code():
    repo_root = Path(__file__).resolve().parents[1]
    runtime_files = [
        repo_root / "scripts" / "demo_offline.py",
        repo_root / "fire_detection_alarm" / "web" / "live.py",
        repo_root / "fire_detection_alarm" / "web" / "pipeline.py",
        repo_root / "configs" / "default.yaml",
    ]

    for file_path in runtime_files:
        assert "cooldown" not in file_path.read_text(encoding="utf-8").lower()
