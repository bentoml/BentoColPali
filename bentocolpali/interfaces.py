from enum import Enum

from pydantic import BaseModel


class ImageDetail(Enum):
    """
    Supported image details for `ImagePayload`.
    """

    LOW = "low"
    HIGH = "high"
    AUTO = "auto"


class ImagePayload(BaseModel):
    """
    Image payload for `ColPaliService`.

    Args:
        url: The URL of the image or a base 64 encoded image.
        detail: The detail of the image. Defaults to `ImageURLDetail.AUTO`.

    NOTE: `detail` is unused for now and only here to be ISO with the OpenAI API.
    """

    url: str
    detail: ImageDetail = ImageDetail.AUTO
