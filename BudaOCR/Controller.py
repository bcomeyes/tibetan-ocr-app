from uuid import UUID
from PySide6.QtCore import QObject, Signal
from BudaOCR.Data import BudaOCRData


class ImageController(QObject):
    on_select_image = Signal(UUID)
    on_receive_image = Signal(UUID)
    def __init__(self):

        super().__init__()

class DataController(QObject):
    on_send_data = Signal(BudaOCRData)
    on_receive_data = Signal(BudaOCRData)
    def __init__(self):
        super().__init__()

