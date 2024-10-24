import base64
from io import BytesIO

from PIL import Image


def is_url(val: str) -> bool:
    return val.startswith("http")


def convert_b64_to_pil_image(b64_image: str) -> Image.Image:
    """
    Convert a base64 image string to a PIL Image.
    """
    if b64_image.startswith("data:image"):
        b64_image = b64_image.split(",")[1]

    try:
        image_data = base64.b64decode(b64_image)
        image_bytes = BytesIO(image_data)
        image = Image.open(image_bytes)
    except Exception as e:
        raise ValueError("Failed to convert base64 string to PIL Image") from e

    return image


def convert_pil_to_b64_image(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 string.
    """
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
