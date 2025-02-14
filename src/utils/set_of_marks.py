from PIL import Image, ImageDraw, ImageFont
from shapely import Polygon, MultiPolygon
from shapely.algorithms import polylabel
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


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = (_COLORS[idx] * maximum).astype(np.uint8)
    if not rgb:
        ret = ret[::-1]
    return ret


def add_num_marks(image: Image.Image, compos: list, fontsize: int = 18) -> Image.Image:
    """
    Adds the number of marks to the image.

    Args:
        image (Image.Image): The image to add the marks to.
        compos (dict): The

    Returns:
        Image.Image: The image with the marks added.
    """
    compos = sorted(compos, key=lambda comp: Polygon(comp["points"]).area, reverse=True)
    non_text_compos = list(filter(lambda comp: comp["class"] != "Text", compos))
    temp_compos = copy.deepcopy(
        non_text_compos
    )  # Prevent changing values in origial list in the next loop
    # If there are more than one compo with the same coords, we keep the one with the longest xpath
    for i, compo in enumerate(temp_compos):
        for j, other_compo in enumerate(temp_compos[:i]):
            if compo["points"] == other_compo["points"]:
                if len(compo["xpath"]) < len(other_compo["xpath"]):
                    non_text_compos.remove(compo)
                else:
                    non_text_compos.remove(other_compo)
                break

    del temp_compos

    mask = Image.new("RGBA", image.size, 0)
    draw_mask = ImageDraw.Draw(mask)

    nums: Image.Image = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(nums)

    font = ImageFont.truetype("arial.ttf", fontsize)

    for compo in non_text_compos:
        compo_color = tuple(random_color(rgb=True, maximum=255))

        lower_compos = get_lower_compos(compo, compos)

        text_location, text_rectangle = _compute_text_location(
            compo, font, lower_compos
        )

        draw_mask.polygon(
            xy=[tuple(point) for point in compo["points"]],
            fill=compo_color + (50,),
        )

        draw_mask.rectangle(
            xy=text_rectangle,
            fill=(0, 0, 0, 200),
        )

        draw.text(
            xy=text_location,
            text=str(compo["id"]),
            fill=compo_color,
            stroke_fill=compo_color,
            stroke_width=0.1,
            font=font,
            anchor="mm",
        )

    image = Image.alpha_composite(image.convert("RGBA"), mask.convert("RGBA"))
    image = Image.alpha_composite(image, nums).convert("RGB")

    image.save("image.png")
    return image


def _compute_text_location(
    compo: dict,
    font: ImageFont.FreeTypeFont,
    lower_compos: list = [],
    surround_factor: float = 0.6,
) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """
    Computes the location and rectangle where the text will be drawn.

    Args:
        compo (dict): The component to draw the text on.
        font (ImageFont): The font to use.
        surround_factor (float): The factor to multiply the text size by to get the surrounding rectangle.

    Returns:
        tuple[tuple[float, float], list[tuple[float, float]]]: The location and rectangle where the text will be drawn.
    """
    compo_mask = Polygon(compo["points"])
    for lower_compo in lower_compos:
        compo_mask = compo_mask.difference(Polygon(lower_compo["points"]))

    _, _, txt_w, txt_h = font.getbbox(str(compo["id"]))
    text_location, text_rectangle = _move_text_to_free_space(
        compo_mask,
        font,
        txt_w,
        txt_h,
        surround_factor,
    )

    # We need to make sure the rectangle does not take too much space of the component
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
        > 0.10
    ):
        # We move it to the top right corner
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
    assert any(
        [isinstance(compo_mask, Polygon), isinstance(compo_mask, MultiPolygon)]
    ), "compo_mask must be a shapely Polygon"

    if isinstance(compo_mask, MultiPolygon):
        compo_mask = max(compo_mask.geoms, key=lambda a: a.area)

    # Find the place in the mask with the most space, then put the text there
    text_location = polylabel.polylabel(compo_mask)
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


def get_lower_compos(compo: dict, other_compos: list) -> list:
    """
    Get the components that are lower than the given component.
    Certain assumptions are made so it is required the components are sorted by area in descending order.

    Args:
        compo (dict): The component to compare to.
        other_compos (list[dict]): The other components to compare to.

    Returns:
        list[dict]: The components that are lower than the given component.
    """
    assert isinstance(compo, dict), "compo must be a dictionary"
    assert "points" in compo, "compo must contain 'points' key"
    assert isinstance(other_compos, list), "other_compos must be a list"

    compo_poly = Polygon(compo["points"])
    lower_compos = []
    for i, comp in enumerate(other_compos):
        if Polygon(comp["points"]).area < compo_poly.area:
            lower_compos.extend(other_compos[i:])
            break
    lower_compos = list(
        filter(
            lambda comp: Polygon(comp["points"]).intersects(compo_poly),
            lower_compos,
        )
    )

    return lower_compos
