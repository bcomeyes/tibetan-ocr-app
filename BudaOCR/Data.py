from uuid import UUID
from enum import Enum
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, List, Tuple


class OpStatus(Enum):
    SUCCESS = 0
    FAILED = 1


class Encoding(Enum):
    Unicode = 0
    Wylie = 1


class ExportFormat(Enum):
    Text = 0
    XML = 1
    JSON = 2


class Theme(Enum):
    Dark = 0
    Light = 1


class LineMode(Enum):
    Line = 0
    Layout = 1


class LineMerge(Enum):
    Merge = 0
    Stack = 1


class LineSorting(Enum):
    Threshold = 0
    Peaks = 1


class TPSMode(Enum):
    GLOBAL = 0
    LOCAL = 1


class Language(Enum):
    English = 0
    German = 1
    French = 2
    Tibetan = 3
    Chinese = 4


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
    center: Tuple[int, int]


@dataclass
class LineData:
    image: npt.NDArray
    prediction: npt.NDArray
    angle: float
    lines: List[Line]


@dataclass
class LayoutData:
    image: npt.NDArray
    rotation: float
    images: List[BBox]
    text_bboxes: List[BBox]
    lines: List[Line]
    captions: List[BBox]
    margins: List[BBox]
    predictions: Dict[str, npt.NDArray]


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
    ocr_text: List[str]
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
    classes: List[str]


@dataclass
class OCRModelConfig:
    model_file: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    charset: List[str]


@dataclass
class LineDataResult:
    guid: UUID
    line_data: LineData


@dataclass
class OCResult:
    guid: UUID
    text: list[str]


@dataclass
class OCRModel:
    guid: UUID
    name: str
    path: str
    config: OCRModelConfig


@dataclass
class OCRSettings:
    line_mode: LineMode
    line_merge: LineMerge
    line_sorting: LineSorting
    dewarping: bool
    tps_mode: TPSMode
    output_encoding: Encoding
    exporters: List[ExportFormat]

@dataclass
class AppSettings:
    model_path: str
    language: Language
    encoding: Encoding
    theme: Theme
