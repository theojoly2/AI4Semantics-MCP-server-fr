from os import mkdir
from os.path import exists
from typing import Any
from pathlib import Path
from json import load
from config import load_config

config = load_config()

MODELS_PATH = Path(config["dir_paths"]["models"])
if not exists(MODELS_PATH):
    mkdir(MODELS_PATH)


def get_model(user: str, session_name: str) -> dict[str, Any]:
    """Returns the user's model for session session_name.

    :param user: The user identifier
    :param session_name: The session name identifier
    :return: Dictionary containing the model data, or empty dict if not found
    """
    file_path = MODELS_PATH / Path(user) / f"{session_name}.json"
    if not exists(file_path):
        return {}

    with open(file_path, "r") as f:
        model: dict[str, Any] = load(f)

    return model
