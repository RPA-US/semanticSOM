from PIL import Image
from io import BytesIO
import base64
import sqlite3
import sqlite_vec
from transformers import CLIPProcessor, CLIPModel
import torch
from src.cfg import CFG
import struct


# Convert Image to Base64
def im_2_b64(image: Image.Image) -> str:
    buff = BytesIO()
    aspect_ratio = image.size[0] / image.size[1]
    max_height = 1080
    max_width = 1920
    if image.size[0] > max_width:
        image = image.resize((max_width, int(max_width / aspect_ratio)))
    if image.size[1] > max_height:
        image = image.resize((int(max_height * aspect_ratio), max_height))
    image.convert("RGB").save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str


def resize_img(image):
    aspect_ratio = image.size[0] / image.size[1]
    max_height = 1080
    max_width = 1920
    if image.size[0] > max_width:
        image = image.resize((max_width, int(max_width / aspect_ratio)))
    if image.size[1] > max_height:
        image = image.resize((int(max_height * aspect_ratio), max_height))
    return image


# Convert Base64 to Image
def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def add_padding(image, horizontal_padding, vertical_padding):
    """
    Add black padding to an image with specified horizontal and vertical padding amounts.
    Places the original image in the center of the new padded image.

    Args:
        image: PIL Image to pad
        horizontal_padding: Padding to add to left and right (total horizontal padding will be 2x this value)
        vertical_padding: Padding to add to top and bottom (total vertical padding will be 2x this value)

    Returns:
        Padded PIL Image
    """
    width, height = image.size
    new_width = width + (horizontal_padding * 2)
    new_height = height + (vertical_padding * 2)

    # Create a new black image with the padded dimensions
    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # Paste the original image in the center
    padded_image.paste(image, (horizontal_padding, vertical_padding))

    return padded_image


class ImageCache:
    """
    A class to handle caching of processed images.
    """

    img_size = (224, 224)  # Standard size for CLIP

    def __init__(self, location: str = CFG.sqlite_db_location) -> None:
        """
        Initializes the Cache class and sets up the SQLite database connection.
        """
        # Instantiate db if not exists
        self.conn: sqlite3.Connection = sqlite3.connect(database=location)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(conn=self.conn)
        self.conn.enable_load_extension(False)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

        self.cursor.execute(
            "CREATE VIRTUAL TABLE image_vectors USING vec0(embedding float[512])"
        )
        self.conn.commit()

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def _serialize_f32(self, vector: list[float]) -> bytes:
        """serializes a list of floats into a compact "raw bytes" format"""
        return struct.pack("%sf" % len(vector), *vector)

    def _vectorize(self, image: Image.Image) -> bytes:
        """Extracts a vector embedding from an image."""
        image = image.resize(self.img_size)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        norm: torch.Tensor = embedding / embedding.norm(
            p=2, dim=-1, keepdim=True
        )  # Normalize
        return self._serialize_f32(norm.squeeze().tolist())

    def in_cache(self, compo_crop: Image.Image) -> bool:
        """
        Checks if the image is in the cache.

        Args:
            compo_crop (Image.Image): The image to check.
        """
        assert isinstance(compo_crop, Image.Image), "compo_crop must be a PIL Image"
        embedding = self._vectorize(compo_crop)

        # Search for similar images in SQLite
        result = self.conn.execute(
            """
            SELECT
                rowid,
                distance
            FROM image_vectors
            WHERE embedding MATCH ?
                AND distance < 0.4
                AND k = 1
            ORDER BY distance
            """,
            (embedding,),
        ).fetchall()

        return result != []

    def update_cache(self, compo_crop: Image.Image) -> None:
        """
        Updates the cache with the image, coordinates, and target element.

        Args:
            compo_crop (Image.Image): The image to cache.
        """
        assert isinstance(compo_crop, Image.Image), "compo_crop must be a PIL Image"
        embedding = self._vectorize(compo_crop)

        self.cursor.execute(
            "INSERT INTO image_vectors (rowid, embedding) VALUES (NULL, ?)",
            (embedding,),
        )
        self.conn.commit()
