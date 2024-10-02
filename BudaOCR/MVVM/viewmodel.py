from uuid import UUID
from typing import List, Dict
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal
from BudaOCR.MVVM.model import BudaOCRDataModel
from BudaOCR.Data import BudaOCRData, LineData


class BudaViewModel(QObject):
    recordChanged = Signal(BudaOCRData)
    dataChanged = Signal(list) # This is actually a list[PalmTreeData] or [], but specifying the type throws errors..
    dataSelected = Signal(BudaOCRData)
    dataCleared = Signal()

    def __init__(self, model: BudaOCRDataModel):
        super().__init__()
        self._model = model

    def add_data(self, data: Dict[UUID, BudaOCRData]):
        self.clear_data()
        self._model.add_data(data)
        current_data = self._model.get_data()
        self.dataChanged.emit(current_data)

    def select_data_by_guid(self, uuid: UUID):
        self.dataSelected.emit(self._model.data[uuid])

    def select_data_by_index(self, index: int):
        current_data = list(self._model.data.values())
        self.dataSelected.emit(current_data[index])

    def get_data_by_guid(self, guid: UUID):
        return self._model.data[guid]

    def update_ocr_data(self, uuid: UUID, text: List[str]):
        self._model.add_ocr_text(uuid, text)
        data = self.get_data_by_guid(uuid)
        self.recordChanged.emit(data)

    def update_page_data(self, uuid: UUID, line_data: LineData, preview_image: npt.NDArray):
        self._model.add_page_data(uuid, line_data, preview_image)
        data = self.get_data_by_guid(uuid)
        self.recordChanged.emit(data)

    def clear_data(self):
        self._model.clear_data()
        self.dataCleared.emit()
