import os
import pyewts
import numpy.typing as npt

from uuid import UUID
from typing import List, Dict
from BDRC.Utils import import_local_models
from BDRC.Data import (
    OCRData,
    Line,
    AppSettings,
    OCRLine,
    OCRLineUpdate,
    OCRSettings,
    OCRModel,
    CharsetEncoder,
)


class SettingsModel:
    def __init__(self, app_settings: AppSettings, ocr_settings: OCRSettings):
        self.app_settings = app_settings
        self.ocr_settings = ocr_settings
        self.ocr_models = []
        self.line_model = ""
        self.layout_model = ""
        self.current_ocr_model = None

        if os.path.isdir(self.app_settings.model_path):
            try:
                ocr_models = import_local_models(self.app_settings.model_path)
                self.ocr_models = ocr_models

                if len(self.ocr_models) > 1:
                    self.current_ocr_model = self.ocr_models[0]
            except BaseException as e:
                pass

    def get_current_ocr_model(self) -> OCRModel | None:
        return self.current_ocr_model

    def update_ocr_settings(self, settings: OCRSettings):
        self.ocr_settings = settings

    def update_app_settings(self, settings: AppSettings):
        self.app_settings = settings

    def set_current_ocr_model(self, ocr_model: OCRModel):
        self.current_ocr_model = ocr_model


class OCRDataModel:
    def __init__(self):
        self.data = {}
        self.converter = pyewts.pyewts()

    def add_data(self, data: Dict[UUID, OCRData]):
        self.data.clear()
        self.data = data

    def get_data(self):
        data = list(self.data.values())
        return data

    def clear_data(self):
        self.data.clear()

    def add_page_data(
        self, guid: UUID, lines: List[Line], preview_image: npt.NDArray, angle: float
    ) -> None:
        self.data[guid].lines = lines
        self.data[guid].preview = preview_image
        self.data[guid].angle = angle

    def add_ocr_text(self, guid: UUID, ocr_lines: List[OCRLine]):
        self.data[guid].ocr_lines = ocr_lines

    def delete_image(self, guid: UUID):
        del self.data[guid]

    def convert_wylie_unicode(self, guid: UUID):
        for ocr_line in self.data[guid].ocr_lines:
            if ocr_line.encoder == CharsetEncoder.Wylie:
                new_text = self.converter.toUnicode(ocr_line.text)
                ocr_line.text = new_text
                ocr_line.encoder = CharsetEncoder.Stack
            else:
                new_text = self.converter.toWylie(ocr_line.text)
                ocr_line.text = new_text
                ocr_line.encoder = CharsetEncoder.Wylie

    def update_ocr_line(self, ocr_line_update: OCRLineUpdate):
        for ocr_line in self.data[ocr_line_update.page_guid].lines:
            if ocr_line.guid == ocr_line_update.ocr_line.guid:
                ocr_line.text = ocr_line_update.ocr_line.text
                ocr_line.encoder = ocr_line_update.ocr_line.encoder
