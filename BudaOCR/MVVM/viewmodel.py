from uuid import UUID
from typing import List, Dict
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal
from BudaOCR.MVVM.model import BudaOCRDataModel, BudaSettingsModel
from BudaOCR.Data import BudaOCRData, Line, OCRModel, AppSettings, OCRSettings


class BudaSettingsViewModel(QObject):
    appSettingsChanged = Signal(AppSettings)
    ocrSettingsChanged = Signal(OCRSettings)
    ocrModelsChanged = Signal(list[OCRModel]) # TODO: Why is this somethis working and sometimes not when using List insteand of list
    ocrModelChanged = Signal(OCRModel)
    def __init__(self, model: BudaSettingsModel):
        super().__init__()
        self._model = model

    def get_ocr_models(self):
        return self._model.ocr_models

    def get_current_ocr_model(self):
        return self._model.current_ocr_model

    def get_ocr_settings(self) -> OCRSettings:
        return self._model.ocr_settings

    def get_app_settings(self) -> AppSettings:
        return self._model.app_settings

    def update_ocr_settings(self, settings: OCRSettings):
        self._model.update_ocr_settings(settings)
        self.ocrSettingsChanged.emit(settings)

    def update_app_settings(self, settings: AppSettings):
        self._model.update_app_settings(settings)
        self.appSettingsChanged.emit(settings)

    def update_ocr_models(self, ocr_models: List[OCRModel]):
        self._model.ocr_models = ocr_models
        self.ocrModelsChanged.emit(ocr_models)

    def select_ocr_model(self, ocr_model: OCRModel):
        self._model.set_current_ocr_model(ocr_model)
        self.ocrModelChanged.emit(ocr_model)


class BudaDataViewModel(QObject):
    recordChanged = Signal(BudaOCRData)
    dataChanged = Signal(list) # This is actually a list[PalmTreeData] or [], but specifying the type throws errors..
    dataSelected = Signal(BudaOCRData)
    """
    Note: The dataAutoSelected Signal is a temporary workaround to handle the case of a data record being selected
    via the page switcher in the header, which focuses and scrolls to the respective image in the ImageGallery. 
    This is for time being a separate signal to avoid having a cycling signal when an image get's selected in the ImageGallery
    via seleced_by_guid which would be focused afterwards as well - which is a weird behaviour
    """
    dataAutoSelected = Signal(BudaOCRData)
    dataCleared = Signal()

    def __init__(self, model: BudaOCRDataModel):
        super().__init__()
        self._model = model

    def get_data_by_guid(self, guid: UUID) -> BudaOCRData:
        return self._model.data[guid]

    def get_data(self) -> Dict[UUID, BudaOCRData]:
        return self._model.data


    def add_data(self, data: Dict[UUID, BudaOCRData]):
        self.clear_data()
        self._model.add_data(data)
        current_data = self._model.get_data()
        self.dataChanged.emit(current_data)

    def select_data_by_guid(self, uuid: UUID):
        self.dataSelected.emit(self._model.data[uuid])

    def get_data_index(self, uuid: UUID):
        _entries = list(self._model.data.keys())
        return _entries.index(uuid)

    def select_data_by_index(self, index: int):
        # This is the case when an index is fed by the PageSwitcher
        current_data = list(self._model.data.values())
        self.dataAutoSelected.emit(current_data[index])

    def update_ocr_data(self, uuid: UUID, text: List[str], silent: bool = False):
        self._model.add_ocr_text(uuid, text)

        if not silent:
            data = self.get_data_by_guid(uuid)
            self.recordChanged.emit(data)

    def update_page_data(self, uuid: UUID, lines: List[Line], preview_image: npt.NDArray, silent: bool = False):
        self._model.add_page_data(uuid, lines, preview_image)

        if not silent:
            data = self.get_data_by_guid(uuid)
            self.recordChanged.emit(data)

    def clear_data(self):
        self._model.clear_data()
        self.dataCleared.emit()