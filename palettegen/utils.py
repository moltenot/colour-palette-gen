import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ImageDetails:
    width: int  # width of the raw image in pixels
    height: int  # height of the raw image in pixels
    file_name: str  # full name of the file
    colours: list[list[int]]  # list of RGB colours (in range 0-255)
    frequencies: list[float]  # floats of the same length as the colours, of which the floats all add to 1
    thumbnail_name: str  # path to pixel thumbnail
    svg_thumbnail_name: str  # path to svg thumbnail

    def to_json(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'file_name': self.file_name,
            'colours': self.colours,
            'frequencies': self.frequencies,
            'thumbnail_name': self.thumbnail_name,
            'svg_thumbnail_name': self.svg_thumbnail_name,
        }


class ImageAnalyzer(ABC):
    image_path: str
    file_name: str  # file name without extension

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.file_name = os.path.basename(self.image_path)
        pass

    @abstractmethod
    def get_image_details(self) -> ImageDetails:
        pass
