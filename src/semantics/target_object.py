import re
import json
import sqlite3
from typing import Any

import polars as pl
import sqlite_vec
from PIL import Image

from src.cfg import CFG
from src.models.models import QwenVLModel
from src.utils.prompt_processing import Coords, process_image_for_prompt


class Cache:
    """
    A class to handle caching of processed images and their target elements.
    """

    def __init__(self) -> None:
        """
        Initializes the Cache class and sets up the SQLite database connection.
        """
        # Instantiate db if not exists
        self.conn: sqlite3.Connection = sqlite3.connect(database=CFG.sqlite_db_location)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(conn=self.conn)
        self.conn.enable_load_extension(False)

    def in_cache(self, img: Image.Image) -> bool | dict[tuple[int, int], str]:
        """
        Checks if the image is in the cache.

        Args:
            img (Image.Image): The image to check.

        Returns:
            bool | dict[tuple[int, int], str]: False if not in cache, otherwise a dictionary of coordinates and target elements.
        """
        assert isinstance(img, Image.Image), "img must be a PIL Image"
        # TODO
        # We first have to set the img to a standard size
        # We also make the img black and white to reduce dimensions and color difference errors
        # Then we can hash it and check if it's in the db

        # img_copy: Image.Image = img.copy()
        # img_copy = img_copy.convert(mode="L")
        # img_copy.thumbnail((1080, 720))
        # img_copy_np: np.ndarray[Any, np.dtype[Any]] = np.array(object=img_copy)

        return False

    def update_cache(
        self, img: Image.Image, coords: Coords, target_element: str
    ) -> None:
        """
        Updates the cache with the image, coordinates, and target element.

        Args:
            img (Image.Image): The image to cache.
            coords (Coords): The coordinates of the target point.
            target_element (str): The identified target element.
        """
        assert isinstance(img, Image.Image), "img must be a PIL Image"
        assert isinstance(coords, Coords), "coords must be an instance of Coords"
        assert isinstance(target_element, str), "target_element must be a string"
        # TODO
        pass


def identify_target_element(
    screenshot: Image.Image, som: dict, coords: Coords, model: Any
) -> str:
    """
    Identifies the target element in the screenshot using the provided model.

    Args:
        screenshot (Image.Image): The screenshot containing the target element.
        som (dict): The structure of the image.
        coords (Coords): The coordinates of the target point.
        model (Any): The model to use for identification.

    Returns:
        str: The identified target element.
    """
    assert isinstance(screenshot, Image.Image), "screenshot must be a PIL Image"
    assert isinstance(som, dict), "som must be a dictionary"
    assert isinstance(coords, Coords), "coords must be an instance of Coords"
    image, sys_prompt, prompt = process_image_for_prompt(
        image=screenshot, som=som, coords=coords
    )

    model_output: str = model(prompt=prompt, sys_prompt=sys_prompt, image=image)
    assert isinstance(model_output, str), "model_output must be a string"

    if match_group := re.search(
        pattern=r"<\|target_element\|>(.*)<\|end_target_element\|>", string=model_output
    ):
        return match_group.group(1)
    raise ValueError("No target element found in the model output.")


def semantize_targets(
    event_log: pl.DataFrame, cache: Cache, model: Any
) -> pl.DataFrame:
    """
    Semantizes the targets in the event log.

    Args:
        event_log (pl.DataFrame): The event log to process.
        cache (Cache): The cache to use for storing processed images.
        model (Any): The model to use for identification.

    Returns:
        pl.DataFrame: The semantized event log.
    """
    assert isinstance(event_log, pl.DataFrame), "event_log must be a Polars DataFrame"
    assert isinstance(cache, Cache), "cache must be an instance of Cache"
    event_target_col: list[str] = []

    try:
        for row in event_log.iter_rows(named=True):
            screenshot: Image.Image = Image.open(
                fp=f"{CFG.image_dir}/{row[CFG.colnames['Screenshot']]}"
            )
            som: dict = json.load(
                open(
                    file=f"{CFG.som_dir}/{row[CFG.colnames['Screenshot']].split('.')[0]}_som.json"
                )
            )

            coords: Coords
            if row[CFG.colnames["Coords"]] and row[CFG.colnames["Coords"]] != "":
                coords = Coords(
                    *map(lambda x: int(x), row[CFG.colnames["Coords"]].split(","))
                )
            else:  # Keyboard event most probably. No way to know at this stage the target element
                event_target_col.append("")
                continue

            if (
                cache_hit := cache.in_cache(img=screenshot)
            ):  # Image already processed. We try to get the target element from the cache
                if coords in cache_hit.keys():  # type: ignore
                    # TODO: Element might be some pixels off, we need a thresshold
                    event_target_col.append(cache_hit[coords])  # type: ignore
                    continue

            event_target_col.append(
                identify_target_element(
                    screenshot=screenshot, som=som, coords=coords, model=model
                )
            )

            cache.update_cache(
                img=screenshot, coords=coords, target_element=event_target_col[-1]
            )
    except Exception as e:  # early stopping if an error occurs
        print(
            f"The following error ocurred while extracting object semantics. Stopping early: {e}"
        )

    event_log = event_log.with_columns(EventTarget=pl.Series(values=event_target_col))
    assert isinstance(event_log, pl.DataFrame), "event_log must be a Polars DataFrame"

    return event_log


if __name__ == "__main__":
    event_log: pl.DataFrame = pl.read_csv(
        source=f"{CFG.project_root}/input/eval/eval.csv"
    )
    model = QwenVLModel(model_name="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")
    event_log = semantize_targets(event_log=event_log, cache=Cache(), model=model)
    event_log.write_csv(file=f"{CFG.project_root}/output/eval.csv")
