from uuid import UUID
from enum import Enum
import numpy.typing as npt
from dataclasses import dataclass

class OpStatus(Enum):
    SUCCESS = 0
    FAILED = 1

class Encoding(Enum):
    Unicode = 0
    Wylie = 1

class LineMode(Enum):
    Line = 0
    Layout = 1

@dataclass
class ScreenData:
    max_width: int
    max_height: int
    start_width: int
    start_height: int
    start_x: int
    start_y: int

@dataclass
class OCRSettings:
    k_factor: float

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class Line:
    contour: npt.NDArray
    bbox: BBox
    center: tuple[int, int]

@dataclass
class LineData:
    image: npt.NDArray
    prediction: npt.NDArray
    angle: float
    lines: list[Line]

@dataclass
class ThemeData:
    name: str
    NewButton: str
    ImportButton: str
    SaveButton: str
    RunButton: str
    SettingsButton: str

@dataclass
class BudaOCRData:
    guid: UUID
    image_path: str
    image_name: str
    ocr_text: list[str]
    line_data: LineData | None
    preview: npt.NDArray | None

@dataclass
class LineDetectionConfig:
    model_file: str
    patch_size: int


@dataclass
class LayoutDetectionConfig:
    model_file: str
    patch_size: int
    classes: list[str]


@dataclass
class OCRConfig:
    model_file: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    charset: list[str]

@dataclass
class LineDataResult:
    guid: UUID
    line_data: LineData

@dataclass
class OCRResult:
    guid: UUID
    text: list[str]