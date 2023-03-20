import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ImageDetails:
    width: int
    height: int
    file_name: str
    colours: list[list[int]]
    frequencies: list[float]
    thumbnail: str  # SVG

    def to_json(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'file_name': self.file_name,
            'colours': self.colours,
            'frequencies': self.frequencies,
            'thumbnail': self.thumbnail,
        }

class ImageAnalyzer(ABC):
    image_path: str
    file_name: str  # file name without extension

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.file_name = os.path.basename(self.image_path).split('.')[0]
        pass

    @abstractmethod
    def get_image_details(self) -> ImageDetails:
        pass
