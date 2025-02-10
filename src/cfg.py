import os


class CFG:
    project_root: str = os.path.dirname(os.path.dirname(__file__))
    image_dir: str = f"{project_root}/images"

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
