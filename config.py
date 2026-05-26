import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_CONFIG = {
    "model_path": "./runs/dog_breed_classifier_cpu.pth",
    "device": "cpu",
    "test_images_dir": "./test_photo",
    "example_images": [
        "./test_photo/img.png",
        "./test_photo/img_1.png"
    ],
    "dataset_root": "./dog_breeds_dataset",
    "output_dir": "./runs",
}


def load_config(config_path: str | None = None) -> dict:
    config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    if yaml is None:
        raise ImportError("PyYAML is required to load config.yaml. Install with `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in data.items() if v is not None})
    return config


if __name__ == "__main__":
    import json
    cfg = load_config()
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
