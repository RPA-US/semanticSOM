from PIL import Image, ImageDraw, ImageFont
from shapely import Polygon, MultiPolygon
from shapely.algorithms import polylabel
from typing import Optional
import numpy as np
import copy


# fmt: off
# RGB:
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        1.000, 1.000, 1.000
    ]
)
_COLORS = _COLORS.astype(np.float16).reshape(-1, 3)
# fmt: on

# Constants
_COLOR_MAX_VALUES = (1, 255)


def random_color(rgb: bool = False, maximum: int = 255) -> np.ndarray:
    """
    Generate a random color vector.

    Args:
        rgb (bool): Whether to return RGB colors (True) or BGR colors (False).
        maximum (int): Maximum color value (either 1 or 255).

    Returns:
        np.ndarray: A vector of 3 numbers representing the color.
    """
    if maximum not in _COLOR_MAX_VALUES:
        raise ValueError(f"maximum must be one of {_COLOR_MAX_VALUES}")

    idx = np.random.randint(0, len(_COLORS))
    color = (_COLORS[idx] * maximum).astype(np.uint8)

    return color if rgb else color[::-1]


def add_num_marks(
    image: Image.Image, compos: list[dict], fontsize: int = 18
) -> Image.Image:
    """
    Adds numbered marks to the image.

    Args:
        image (Image.Image): The image to add the marks to.
        compos (list[dict]): List of component dictionaries.
        fontsize (int): Font size for the marks.

    Returns:
        Image.Image: The image with the marks added.
    """
    if fontsize < 1:
        raise ValueError("fontsize must be greater than 0")

    compos = list(
        filter(
            lambda comp: Polygon(comp["points"]).is_valid
            and not Polygon(comp["points"]).is_empty,
            compos,
        )
    )
    non_text_compos = _filter_and_sort_compos(compos)

    mask, nums = _prepare_layers(image.size)
    font = ImageFont.truetype("arial.ttf", fontsize)

    for compo in non_text_compos:
        compo_color = tuple(random_color(rgb=True))
        lower_compos = get_lower_compos(compo, compos)
        text_location, text_rectangle = _compute_text_location(
            compo, font, lower_compos
        )

        if text_location is not None:
            _draw_component(
                mask, nums, compo, text_location, text_rectangle, compo_color, font
            )

    image = Image.alpha_composite(image.convert("RGBA"), mask.convert("RGBA"))
    image = Image.alpha_composite(image, nums).convert("RGB")

    return image


def _filter_and_sort_compos(compos: list[dict]) -> list[dict]:
    """
    Filter non-text components and sort them by area in descending order.
    Also removes duplicate components with shorter xpaths.

    Args:
        compos (list[dict]): List of component dictionaries.

    Returns:
        list[dict]: Filtered and sorted list of non-text components.
    """
    compos = sorted(compos, key=lambda comp: Polygon(comp["points"]).area, reverse=True)
    non_text_compos = [comp for comp in compos if comp["class"] != "Text"]

    temp_compos = copy.deepcopy(non_text_compos)

    for i, compo in enumerate(temp_compos):
        for other_compo in temp_compos[:i]:
            if compo["points"] == other_compo["points"]:
                if (
                    len(compo["xpath"]) < len(other_compo["xpath"])
                    and compo in non_text_compos
                ):
                    non_text_compos.remove(compo)
                elif other_compo in non_text_compos:
                    non_text_compos.remove(other_compo)
                break

    return non_text_compos


def _prepare_layers(image_size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    """
    Prepare the mask and number layers for drawing.

    Args:
        image_size (tuple[int, int]): Size of the image.

    Returns:
        tuple[Image.Image, Image.Image]: The mask and number layers.
    """
    mask = Image.new("RGBA", image_size, 0)
    nums = Image.new("RGBA", image_size, (255, 255, 255, 0))
    return mask, nums


def _draw_component(
    mask, nums, compo, text_location, text_rectangle, compo_color, font
):
    """
    Draws the component and its number on the mask and number layers.
    """
    draw_mask = ImageDraw.Draw(mask)
    draw_nums = ImageDraw.Draw(nums)

    draw_mask.polygon(
        xy=[tuple(point) for point in compo["points"]],
        fill=compo_color + (50,),
        outline=compo_color + (255,),
    )

    draw_mask.rectangle(xy=text_rectangle, fill=(0, 0, 0, 200))

    draw_nums.text(
        xy=text_location,
        text=str(compo["id"]),
        fill=compo_color,
        stroke_fill=compo_color,
        stroke_width=0.1,
        font=font,
        anchor="mm",
    )


def _compute_text_location(
    compo: dict,
    font: ImageFont.FreeTypeFont,
    lower_compos: list = [],
    surround_factor: float = 0.6,
    allowed_overshadowing: float = 0.10,
) -> tuple[Optional[tuple[float, float]], Optional[list[tuple[float, float]]]]:
    """
    Computes the location and rectangle where the text will be drawn.
    """
    compo_mask = Polygon(compo["points"])
    for lower_compo in lower_compos:
        compo_mask = compo_mask.difference(Polygon(lower_compo["points"]))
    if compo_mask.is_empty:
        return None, None

    _, _, txt_w, txt_h = font.getbbox(str(compo["id"]))
    try:
        text_location, text_rectangle = _move_text_to_free_space(
            compo_mask,
            font,
            txt_w,
            txt_h,
            surround_factor,
        )
    except TimeoutError:
        return None, None

    if (
        compo_mask.intersection(
            Polygon(
                [
                    text_rectangle[0],
                    (text_rectangle[0][0], text_rectangle[1][1]),
                    text_rectangle[1],
                    (text_rectangle[1][0], text_rectangle[0][1]),
                ]
            )
        ).area
        / compo_mask.area
        > allowed_overshadowing
    ):
        text_location = (
            text_location[0] + txt_w * surround_factor,
            text_location[1] - txt_h * surround_factor,
        )
        text_rectangle = [
            (
                text_location[0] - txt_w * surround_factor,
                text_location[1] - txt_h * surround_factor,
            ),
            (
                text_location[0] + txt_w * surround_factor,
                text_location[1] + txt_h * surround_factor,
            ),
        ]

    return text_location, text_rectangle


def _move_text_to_free_space(
    compo_mask: Polygon,
    font: ImageFont.FreeTypeFont,
    txt_w: float,
    txt_h: float,
    surround_factor: float = 0.6,
) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """
    Moves the text to a free space.

    Args:
        text_location (tuple[float, float]): The location of the text.
        text_rectangle (list[tuple[float, float]]): The rectangle where the text will be drawn.
        compo_mask (Polygon): The mask of the component.

    Returns:
        tuple[tuple[float, float], list[tuple[float, float]]]: The new location and rectangle where the text will be drawn.
    """

    def timeout_hander(signum, frame):
        raise TimeoutError("Timeout reached")

    assert any(
        [isinstance(compo_mask, Polygon), isinstance(compo_mask, MultiPolygon)]
    ), "compo_mask must be a shapely Polygon"

    if isinstance(compo_mask, MultiPolygon):
        compo_mask = max(compo_mask.geoms, key=lambda a: a.area)

    # Find the place in the mask with the most space, then put the text there
    # signal.signal(signal.SIGALRM, timeout_hander)
    # signal.alarm(2)
    text_location = polylabel.polylabel(compo_mask)
    # signal.alarm(0)
    text_rectangle = [
        (
            text_location.x - txt_w * surround_factor,
            text_location.y - txt_h * surround_factor,
        ),
        (
            text_location.x + txt_w * surround_factor,
            text_location.y + txt_h * surround_factor,
        ),
    ]

    return (text_location.x, text_location.y), text_rectangle


def get_lower_compos(compo: dict, other_compos: list[dict]) -> list[dict]:
    """
    Get the components that are lower than the given component.
    Assumes components are sorted by area in descending order.
    """
    compo_poly = Polygon(compo["points"])
    lower_compos = [
        c
        for c in other_compos
        if Polygon(c["points"]).area < compo_poly.area
        and Polygon(c["points"]).intersects(compo_poly)
    ]
    return lower_compos
