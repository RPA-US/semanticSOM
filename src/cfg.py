import os


class CFG:
    """
    Configuration class for the project.
    """

    project_root: str = os.path.dirname(os.path.dirname(__file__))
    image_dir: str = f"{project_root}/input/images"
    som_dir: str = f"{project_root}/input/soms"

    sqlite_db_location: str = f"{project_root}/cache.db"

    colnames: dict[str, str] = {
        "Coords": "Coords",
        "ScreenID": "ScreenID",
        "Screenshot": "Screenshot",
        "EventType": "EventType",
        "EventTarget": "EventTarget",
        "Text": "Text",
        "EventDescription": "EventDescription",
        "Activity": "Activity",
        "ActvityDescription": "ActvityDescription",
    }

    prompt_config: dict[str, list[str] | str] = {
        "technique": ["highlight"],  # possible values: "som", "highlight"
        "crop": "parent",  # possible values: "parent", "target", "none"
    }
