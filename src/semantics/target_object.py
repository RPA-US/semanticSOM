from PIL import Image
import json
import polars as pl
from src.cfg import CFG
from typing import NamedTuple, Any
import numpy as np
import sqlite3
import sqlite_vec
from src.models.models import QwenVLModel


class Coords(NamedTuple):
    x: int
    y: int


class Cache:
    def __init__(self) -> None:
        # Instantiate db if not exists
        self.conn: sqlite3.Connection = sqlite3.connect(database=CFG.sqlite_db_location)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(conn=self.conn)
        self.conn.enable_load_extension(False)

    def process_img(self, img: Image.Image) -> str:
        """
        Computes the hash of the image and returns it as a vector to allow for image similarity search

        Uses whash to compute the hash of the image
        """
        img = img.convert(mode="L")
        img.thumbnail((1080, 720))
        img = Image.hash_whash(img)
        return img

    def in_cache(self, img: Image.Image) -> bool | dict[tuple[int, int], str]:
        # TODO
        # We first have to set the img to a standard size
        # We also make the img black and white to reduce dimensions and color difference errors
        # Then we can hash it and check if it's in the db

        img_copy: Image.Image = img.copy()
        img_copy = img_copy.convert(mode="L")
        img_copy.thumbnail((1080, 720))
        img_copy_np: np.ndarray[Any, np.dtype[Any]] = np.array(object=img_copy)

        return False

    def update_cache(
        self, img: Image.Image, coords: Coords, target_element: str
    ) -> None:
        # TODO
        pass


def identify_target_element(
    screenshot: Image.Image, som: dict, coords: Coords, model: Any
) -> str:
    # TODO

    return ""


def semantize_targets(
    event_log: pl.DataFrame, cache: Cache, model: Any
) -> pl.DataFrame:
    event_target_col: list[str] = []

    for row in event_log.iter_rows(named=True):
        screenshot: Image.Image = Image.open(
            fp=f"{CFG.image_dir}/{row[CFG.colnames['Screenshot']]}"
        )
        som: dict = json.loads(s=f"{CFG.som_dir}/{row[CFG.colnames['Screenshot']]}")

        coords: Coords
        if row[CFG.colnames["Coords"]] and row[CFG.colnames["Coords"]] != "":
            coords = Coords(
                *map(lambda x: int(x), row[CFG.colnames["Coords"]].split(","))
            )
        else:  # Keyboard event most probably. No way to know at this stage the target element
            event_target_col.append("")
            continue

        if cache_hit := cache.in_cache(
            img=screenshot
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

    return event_log


if __name__ == "__main__":
    event_log: pl.DataFrame = pl.read_csv(source=f"{CFG.project_root}/input/email.csv")
    model = QwenVLModel(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    event_log = semantize_targets(event_log=event_log, cache=Cache(), model=model)
    event_log.write_csv(file=f"{CFG.project_root}/output/email.csv")
