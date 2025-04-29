import os


class CFG:
    """
    Configuration class for the project.
    """

    project_root: str = os.path.dirname(os.path.dirname(__file__))
    image_dir: str = f"{project_root}/input/images"
    som_dir: str = f"{project_root}/input/soms"

    s2s_dataset_dir: str = f"{project_root}/input/screen2som_dataset"

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
        "GroundTruth": "GroundTruth",
    }

    prompt_config: dict[str, list[str] | str | bool] = {
        "cot": True,
        "technique": ["som"],  # possible values: "som", "highlight"
        "crop": "parent",  # possible values: "parent", "target", "none"
    }

    eval_config: dict[str, str] = {
        "jsonl_path": f"{project_root}/input/eval/eval.jsonl",
        "model": "local_qwen_vl",
        "output_path": f"{project_root}/output/eval_result.jsonl",
    }
