
from pathlib import Path

import yaml


def load_config() -> dict:
    """Load the application configuration from config.yaml."""
    config_path = (
        Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    
    if not isinstance(config, dict):
        raise TypeError(
            f"Invalid configuration format in {config_path}"
        )

    return config

