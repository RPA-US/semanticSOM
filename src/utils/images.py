from PIL import Image
from io import BytesIO
import base64


# Convert Image to Base64
def im_2_b64(image):
    buff = BytesIO()
    image.convert("RGB").save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str


# Convert Base64 to Image
def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)
