import yaml
from pathlib import Path

def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)