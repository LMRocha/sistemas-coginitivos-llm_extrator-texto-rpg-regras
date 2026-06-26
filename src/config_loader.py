# ROOT/src/config_loader.py
import yaml
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration.

    Args:
        config_path: Optional explicit path. If None, uses
                     ROOT/configs/config.yaml relative to this module.

    Returns:
        Dictionary with configuration keys.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if config_path is None:
        # __file__ is the path to this file (src/config_loader.py)
        # .parent -> src/, .parent.parent -> ROOT/
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)