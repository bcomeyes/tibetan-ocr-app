from uuid import UUID
import numpy.typing as npt
from typing import List, Dict
from BudaOCR.Data import BudaOCRData, LineData


class BudaOCRDataModel:
    def __init__(self):
        self.data = {}

    def add_data(self, data: Dict[UUID, BudaOCRData]):
        self.data.clear()
        self.data = data

    def get_data(self):
        data = list(self.data.values())
        return data

    def clear_data(self):
        self.data.clear()

    def add_page_data(self, guid: UUID, line_data: LineData, preview_image: npt.NDArray) -> None:
        self.data[guid].line_data = line_data
        self.data[guid].preview = preview_image

    def add_ocr_text(self, guid: UUID, text: List[str]):
        self.data[guid].ocr_text = text

