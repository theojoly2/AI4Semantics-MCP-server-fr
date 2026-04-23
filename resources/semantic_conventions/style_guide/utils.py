from os.path import exists
from pathlib import Path
from pathlib import Path
from json import load
from config import load_config

config = load_config()

STYLE_GUIDE_TXT_PATH = Path(config["file_paths"]["style_guide"])

async def get_style_guide() -> str:
    """Returns the style guide content.

    :return: String containing the style guide content, or error message if not found
    """
    file_path = STYLE_GUIDE_TXT_PATH
    if not exists(file_path):
        return "No such documentation available"

    with open(file_path, "r") as f:
        resource: str = f.read()

    return resource
