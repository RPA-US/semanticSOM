from PIL import Image, ImageDraw, ImageFont
from shapely import Polygon
import numpy as np

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
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
) * 255
_COLORS = _COLORS.astype(int).reshape(-1, 3)
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
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def add_num_marks(image: Image.Image, compos: list) -> Image.Image:
    """
    Adds the number of marks to the image.

    Args:
        image (Image.Image): The image to add the marks to.
        compos (dict): The

    Returns:
        Image.Image: The image with the marks added.
    """
    compos = sorted(compos, key=lambda comp: Polygon(comp["points"]).area, reverse=True)

    mask = Image.new("RGB", image.size, 0)
    draw_mask = ImageDraw.Draw(mask)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for compo in compos:
        compo_color = tuple(random_color(rgb=True, maximum=255))
        text_location = compo["centroid"]
        # compute the rectangle where text will be drawn usitng its location and font info
        _, _, txt_w, txt_h = font.getbbox(str(compo["id"]))
        text_rectangle: list[tuple[float, float]] = [
            (text_location[0] - txt_w, text_location[1] - txt_h),
            (text_location[0] + txt_w, text_location[1] + txt_h),
        ]

        draw_mask.polygon(
            xy=[tuple(point) for point in compo["points"]],
            fill=compo_color,
        )

        draw.rectangle(
            xy=text_rectangle,
            fill="black",
        )

        draw.text(
            xy=tuple(text_location),
            text=str(compo["id"]),
            fill=compo_color,
            font=font,
            anchor="ms",
        )

    return image
