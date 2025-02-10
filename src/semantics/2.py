from PIL import Image
import polars as pl
from src.cfg import CFG
from typing import NamedTuple
import sqlite3
import sqlite_vec


class Coords(NamedTuple):
    x: int
    y: int


class Cache:
    def __init__(self):
        # Instantiate db if not exists
        self.conn = sqlite3.connect(CFG.sqlite_db_location)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def in_cache(self, img: Image.Image) -> bool | dict[tuple[int, int], str]:
        # TODO
        # We first have to set the img to a standard size
        # We also make the img black and white to reduce dimensions and color difference errors
        # Then we can hash it and check if it's in the db

        return False

    def update_cache(
        self, img: Image.Image, coords: Coords, target_element: str
    ) -> None:
        # TODO
        pass


def identify_target_element(screenshot: Image.Image, coords: Coords) -> str:
    # TODO
    return ""


def semantize_targets(event_log: pl.DataFrame, cache: Cache) -> pl.DataFrame:
    event_target_col: list[str] = []

    for row in event_log.iter_rows(named=True):
        screenshot: Image.Image = Image.open(
            f"{CFG.project_root}/input/{row[CFG.colnames['Screenshot']]}"
        )

        coords: Coords
        if row[CFG.colnames["Coords"]] and row[CFG.colnames["Coords"]] != "":
            coords = Coords(
                *map(lambda x: int(x), row[CFG.colnames["Coords"]].split(","))
            )
        else:  # Keyboard event most probably. No way to know at this stage the target element
            event_target_col.append("")
            continue

        if cache_hit := cache.in_cache(
            screenshot
        ):  # Image already processed. We try to get the target element from the cache
            if coords in cache_hit.keys():  # type: ignore
                # TODO: Element might be some pixels off, we need a thresshold
                event_target_col.append(cache_hit[coords])  # type: ignore
                continue

        event_target_col.append(identify_target_element(screenshot, coords))

        cache.update_cache(screenshot, coords, event_target_col[-1])

    return event_log


if __name__ == "__main__":
    event_log = pl.read_csv(f"{CFG.project_root}/input/email.csv")
    event_log = semantize_targets(event_log, Cache())
    event_log.write_csv(f"{CFG.project_root}/output/email.csv")
