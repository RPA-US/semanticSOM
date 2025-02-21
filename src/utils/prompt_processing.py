from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from src.cfg import CFG
from src.semantics.prompts import (
    COT_ACTION_TARGET_BASE,
    COT_ACTION_TARGET_COORDS,
    COT_ACTION_TARGET_ELEM,
)
from src.utils.set_of_marks import add_num_marks


class Coords(NamedTuple):
    x: int
    y: int


def hash_image(image: Image.Image) -> None:
    """
    Computes the hash of the image and returns it as a vector to allow for image similarity search.

    Uses whash to compute the hash of the image.

    Args:
        image (Image.Image): The image to hash.
    """
    pass


def identify_target_object(image: Image.Image, som: dict, coords: Coords) -> dict:
    """
    Identifies the target object in the image based on the provided SOM and coordinates.
    What we look for is the smallest component that contains the point.

    Args:
        image (Image.Image): The image containing the target object.
        som (dict): The structure of the image.
        coords (Coords): The coordinates of the target point.

    Returns:
        dict: The identified target object.
    """
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    assert isinstance(som, dict), "som must be a dictionary"
    assert isinstance(coords, Coords), "coords must be an instance of Coords"

    target_object: dict = dict()

    non_text_compos = list(
        filter(lambda compo: compo["class"] != "Text", som["compos"])
    )

    for compo in non_text_compos:
        poly = Polygon(compo["points"])
        point = Point(coords.x, coords.y)
        if poly.contains(point) and (
            not target_object or poly.area < Polygon(target_object["points"]).area
        ):
            target_object = compo

    if not target_object:
        text_compos = list(
            filter(lambda compo: compo["class"] == "Text", som["compos"])
        )
        for compo in text_compos:
            poly = Polygon(compo["points"])
            point = Point(coords.x, coords.y)
            if poly.contains(point) and (
                not target_object or poly.area < Polygon(target_object["points"]).area
            ):
                target_object = compo

    assert isinstance(target_object, dict), "target_object must be a dictionary"
    return target_object


def bbox_from_object(obj: dict) -> tuple[int, int, int, int]:
    """
    Extract the upper leftmost, lower rightmost coordinates of the object from its list of points.

    Args:
        obj (dict): The object to extract the bounding box from.

    Returns:
        tuple[int, int, int, int]: The bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    assert isinstance(obj, dict), "obj must be a dictionary"
    assert "points" in obj, "obj must contain 'points' key"

    x_coords = [point[0] for point in obj["points"]]
    y_coords = [point[1] for point in obj["points"]]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def process_image_for_prompt(
    image: Image.Image, som: dict, coords: Coords
) -> tuple[Image.Image, str, str]:
    """
    Constructs the image for the prompt based on configuration parameters.

    Possible configurations include:
    - Set of Marks / Markers to be displayed on the image
    - Highlighting of target elements
    - Use of the full image
    - Use of a cropped image (parent of target object)
    - Use of a cropped image (target object)

    Args:
        image (Image.Image): The image to process.
        som (dict): The structure of the image.
        coords (Coords): The coordinates of the target point.

    Returns:
        tuple[Image.Image, str, str]: The processed image, system prompt, and user prompt.
    """
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    assert isinstance(som, dict), "som must be a dictionary"
    assert isinstance(coords, Coords), "coords must be an instance of Coords"

    target_object: dict = identify_target_object(image=image, som=som, coords=coords)
    sys_prompt: str = COT_ACTION_TARGET_BASE
    prompt: str = ""
    img_shape = image.size

    match CFG.prompt_config["crop"]:
        case "parent":
            parent_id = target_object["xpath"][-2]
            parent_object = next(
                filter(lambda compo: compo["id"] == parent_id, som["compos"]), None
            )
            image, som["compos"] = (
                crop_and_adjust_bbox(
                    image=image,
                    compos=som["compos"],
                    crop_coords=bbox_from_object(obj=parent_object),
                )
                if parent_object
                else (image, som["compos"])
            )
        case "target":
            image, som["compos"] = crop_and_adjust_bbox(
                image=image,
                compos=som["compos"],
                crop_coords=bbox_from_object(obj=target_object),
            )
            sys_prompt = COT_ACTION_TARGET_ELEM
            prompt = ""  # Intentionally blank
        case _:
            pass

    if CFG.prompt_config["crop"] != "target":
        if "highlight" in CFG.prompt_config["technique"]:
            # Highlight the target object
            image = highlight_compo(image=image, compo=target_object)
            prompt = "Identify the object highlighted in the image."
        if "som" in CFG.prompt_config["technique"]:
            # We recompute the ids to make it easir to identify the target object (lower numbers make it easier, same context and info, less cognitive load)
            for idx, compo in enumerate(
                filter(lambda c: c["class"] != "Text", som["compos"])
            ):
                compo["id"] = idx + 1

            # Add the SOM to the image
            image = add_num_marks(
                image=image,
                compos=som["compos"],
            )
            prompt = (
                f"Identify the element marked in the image with the number {target_object['id']}"  # DANGER: If target object becomes a non-shallow copy, this id will be the old one
                if not prompt
                else prompt
                + f" Note that the element is identified with the number {target_object['id']}."
            )
        if CFG.prompt_config["technique"] == []:
            sys_prompt = COT_ACTION_TARGET_COORDS
            new_img_shape = image.size
            new_coords = Coords(
                x=coords.x * new_img_shape[0] // img_shape[0],
                y=coords.y * new_img_shape[1] // img_shape[1],
            )
            prompt = (
                f"Identify the element at coordinates ({new_coords.x}, {new_coords.y})"
            )

    assert isinstance(image, Image.Image), "processed image must be a PIL Image"
    assert isinstance(sys_prompt, str), "sys_prompt must be a string"
    assert isinstance(prompt, str), "prompt must be a string"
    return image, sys_prompt, prompt


def highlight_compo(image: Image.Image, compo: dict) -> Image.Image:
    """
    Highlights the component in the image.

    Args:
        image (Image.Image): The image containing the component.
        compo (dict): The component to highlight.

    Returns:
        Image.Image: The image with the component highlighted.
    """
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    assert isinstance(compo, dict), "compo must be a dictionary"
    assert "points" in compo, "compo must contain 'points' key"

    cv_img = cv2.cvtColor(src=np.array(object=image), code=cv2.COLOR_RGB2BGR)
    cv2.polylines(
        img=cv_img,
        pts=[np.array(object=compo["points"], dtype=np.int32)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3,
    )
    return Image.fromarray(obj=cv2.cvtColor(src=cv_img, code=cv2.COLOR_BGR2RGB))


def crop_and_adjust_bbox(
    image: Image.Image, compos: list, crop_coords: tuple[int, int, int, int]
) -> tuple[Image.Image, list]:
    """
    Crops the image to the bounding box specified and adjust all the components coords accordingly.

    Args:
        image (Image.Image): The image to crop.
        som (dict): The structure of the image.
        crop_coords (tuple[int, int]): The coordinates of the bounding box to crop.

    Returns:
        Image.Image: The cropped image.
    """
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    assert isinstance(compos, list), "compos must be a list"
    assert all(isinstance(compo, dict) for compo in compos), (
        "compos must be a list of dictionaries"
    )

    # 1. Crop the image
    cropped_image = image.crop(box=crop_coords)
    # 2. Filter by components contained in the crop
    remaining_compos = list(
        filter(
            lambda compo: Polygon(compo["points"]).intersects(
                Polygon(
                    [
                        (crop_coords[0], crop_coords[1]),
                        (crop_coords[2], crop_coords[1]),
                        (crop_coords[2], crop_coords[3]),
                        (crop_coords[0], crop_coords[3]),
                    ]
                )
            ),
            compos,
        )
    )
    # 3. Adjust the coordinates of the remaining components
    for compo in remaining_compos:
        compo["points"] = [
            (
                max(
                    0,
                    min(
                        point[0] - crop_coords[0],
                        crop_coords[2] - crop_coords[0],
                    ),
                ),
                max(
                    0,
                    min(
                        point[1] - crop_coords[1],
                        crop_coords[3] - crop_coords[1],
                    ),
                ),
            )
            for point in compo["points"]
        ]
        compo["centroid"] = Polygon(compo["points"]).centroid.coords[0]

    return cropped_image, remaining_compos
