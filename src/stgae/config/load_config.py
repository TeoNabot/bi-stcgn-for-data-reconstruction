from pathlib import Path
import yaml
from stgae.utils.paths import PROJECT_ROOT

def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # resolve paths relative to project root
    if "paths" in config:
        for key, rel_path in config["paths"].items():
            config["paths"][key] = PROJECT_ROOT / rel_path

    return config
