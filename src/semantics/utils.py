from PIL import Image
from src.semantics.target_object import Coords


def hash_image(image: Image.Image) -> str:
    """
    Computes the hash of the image and returns it as a vector to allow for image similarity search

    Uses whash to compute the hash of the image
    """
    img = image.convert(mode="L")
    img.thumbnail((1080, 720))
    img = Image.hash_whash(img)
    return img


def identify_target_object(image: Image.Image, som: dict, coords: Coords) -> dict:
    """
    Identifies the target object in the image based on the provided SOM and coordinates.

    Returns the identified target object
    """
    # TODO
    res: dict = dict()
    return res


def process_image_for_prompt(image: Image.Image, som: dict, coords: Coords) -> dict:
    """
    Constructs the image for the prompt based on configuration parameters.

    Possible configurations include:
    - Set of Marks / Markers to be displayed on the image
    - Highlighting of target elements
    - Use of the full image
    - Use of a cropped image (parent of target object)
    - Use of a cropped image (target object)

    Returns the constructed prompt for the LLM to consume
    """
    # TODO
    target_object = identify_target_object(image=image, som=som, coords=coords)
    res: dict = dict()
    return res
