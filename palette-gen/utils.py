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


class ImageAnalyzer(ABC):
    image_path: str
    file_name: str # file name without extension

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.file_name = os.path.basename(self.image_path).split('.')[0]
        pass

    @abstractmethod
    def to_json(self) -> ImageDetails:
        pass
