from uuid import UUID
import numpy.typing as npt
from typing import List, Dict
from BudaOCR.Utils import import_local_models
from BudaOCR.Data import BudaOCRData, LineData, AppSettings, OCRSettings, OCRModel


class BudaSettingsModel:
    def __init__(self, app_settings: AppSettings, ocr_settings: OCRSettings):
        self.app_settings = app_settings
        self.ocr_settings = ocr_settings
        self.ocr_models = import_local_models(self.app_settings.model_path)
        self.line_model = ""
        self.layout_model = ""
        self.current_ocr_model = self.set_default_ocr_model()

        print(self.current_ocr_model)

    def set_default_ocr_model(self) -> OCRModel | None:
        if len(self.ocr_models) > 0:
            return self.ocr_models[0]
        else:
            return None

    def get_current_ocr_model(self) -> OCRModel | None:
        return self.current_ocr_model

    def update_ocr_settings(self, settings: OCRSettings):
        self.ocr_settings = settings

    def update_app_settings(self, settings: AppSettings):
        self.app_settings = settings

    def set_current_ocr_model(self, ocr_model: OCRModel):
        self.current_ocr_model = ocr_model


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
