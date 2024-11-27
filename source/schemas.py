from typing import List, Any, Optional

from PIL import Image
from pydantic import BaseModel


class AnnotationData(BaseModel):
    class_id: int
    bbox: List[float]
    cropped_image: Optional[Any]


class ImageData(BaseModel):
    name: str
    image: Any
    width: int
    height: int
    annotations: List[AnnotationData]
