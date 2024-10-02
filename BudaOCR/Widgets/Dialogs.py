import uuid
from uuid import UUID

from huggingface_hub import HfApi
from PySide6.QtCore import Qt, QThreadPool, Signal, QSize
from PySide6.QtWidgets import QFileDialog, QMessageBox, QDialog, QLabel, QVBoxLayout, QHBoxLayout, QDialogButtonBox, \
    QProgressDialog, QPushButton, QListWidget, QListView, QListWidgetItem, QWidget

from BudaOCR.Data import BudaOCRData, LineMode, OCRResult, LineDataResult
from BudaOCR.Inference import LayoutDetection, LineDetection
from BudaOCR.Runner import OCRunner


class ImportFilesDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ImportFilesDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.setNameFilter("Images (*.png *.jpg *.tif *.tiff)")
        self.setViewMode(QFileDialog.ViewMode.List)


class ImportDirDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ImportDirDialog, self).__init__(parent)
    #dir_ = QFileDialog.getExistingDirectory(None, options=QFileDialog.Option.ShowDirsOnly)


class NotificationDialog(QMessageBox):
    def __init__(self, title: str, message: str):
        super().__init__()
        self.setObjectName("NotificationWindow")
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self.setMinimumHeight(440)
        self.setIcon(QMessageBox.Icon.Information)
        self.setStandardButtons(QMessageBox.Ok)

        self.setStyleSheet("""

                    QPushButton {
                        width: 200px;
                        padding: 5px;
                        background-color: #4d4d4d;
                    }
                """)
class ModelListWidget(QWidget):
    def __init__(self, guid: UUID, title: str, parent=None):
        super().__init__()
        self.guid = guid
        self.title = str(title)
        self.label = QLabel(self.title)
        self.download_btn = QPushButton('Download')
        self.delete_btn = QPushButton('Delete')

        # build layout
        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.label)
        self.h_layout.addWidget(self.download_btn)
        self.h_layout.addWidget(self.delete_btn)
        self.setLayout(self.h_layout)

        self.setStyleSheet("""
            color: #ffffff;
        """)

class ModelList(QListWidget):
    sign_on_selected_item = Signal(UUID)

    def __init__(self, parent=None):
        super(ModelList, self).__init__(parent)
        self.parent = parent
        self.setObjectName("ModelListItem")
        self.setFlow(QListView.Flow.TopToBottom)
        self.setMouseTracking(True)
        self.itemClicked.connect(self.on_item_clicked)

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(
            item
        )  # returns an instance of CanvasHierarchyEntry

        if isinstance(_list_item_widget, ModelListWidget):
            print(f"Clicked on Model: {_list_item_widget.title}")
            self.sign_on_selected_item.emit(_list_item_widget.guid)


class SettingsWindow(QDialog):
    def __init__(self, hf_api: HfApi):
        super().__init__()
        self.setWindowTitle("BudaOCR Settings")
        self.setFixedHeight(400)
        self.setFixedWidth(600)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.label = QLabel("BudaOCR Settings")
        self.btn_refresh = QPushButton("Refresh")

        self.hf_api = hf_api
        self.online_label = QLabel()
        self.spacer = QLabel()
        self.label.setFixedHeight(32)
        self.model_list = ModelList(self)

        # define layout
        self.top_v_layout = QVBoxLayout()
        self.top_v_layout.addWidget(self.label)
        self.top_v_layout.addWidget(self.btn_refresh)

        self.main_v_layout = QVBoxLayout()
        self.main_v_layout.addLayout(self.top_v_layout)
        self.main_v_layout.addWidget(self.spacer)
        self.main_v_layout.addWidget(self.model_list)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.cancel_btn = QPushButton("Cancel")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)
        self.main_v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.main_v_layout)

        # bind signals
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.setStyleSheet("""
        
            background-color: #1d1c1c;
            color: #ffffff;
        
            QLabel {
                color: #000000;
            }
            QDialogButtonBox::Ok {
                height: 32px;
                width: 64px;
            }
            QDialogButtonBox::Cancel {
                height: 32px;
                width: 64px;
            }
        """)

        try:
            self.models = self.hf_api.list_models(author="BDRC")
            self.online_label = f"Available Models:"
            self.build_model_overview()
        except ConnectionError as e:
            self.online_label = f"No internet connection: {e}. Please load models locally."

    def handle_accept(self):
        self.accept()

    def handle_reject(self):
        self.reject()

    def refresh_models(self):
        try:
            self.models = self.hf_api.list_models(author="BDRC")
            self.build_model_overview()
        except ConnectionError as e:
            self.online_label = f"No internet connection: {e}. Please load models locally."

        finally:
            self.online_label = f"Available Models:"

    def build_model_overview(self):
        self.model_list.clear()
        print(f"Building model overview...")

        for model in self.models:
            print(f"Model: {model.id}")
            model_item = QListWidgetItem(self.model_list)
            model_widget = ModelListWidget(
                guid=uuid.uuid1(),
                title=model.id
            )

            model_item.setSizeHint(model_widget.sizeHint())

            self.model_list.addItem(model_item)
            self.model_list.setItemWidget(model_item, model_widget)

    def clear_models(self):
        self.model_list.clear()

class OCRBatchProgress(QProgressDialog):
    sign_line_result = Signal(LineDataResult)
    sign_ocr_result = Signal(OCRResult)

    def __init__(self, data: list[BudaOCRData], line_detection: LineDetection, layout_detection: LayoutDetection, mode: LineMode, pool: QThreadPool, parent=None):
        super(OCRBatchProgress, self).__init__(parent)
        self.setObjectName("OCRDialog")
        self.setMinimumWidth(300)
        self.setWindowTitle("OCR Progress")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimum(0)
        self.setMaximum(0)

        self.data = data
        self.line_detection = line_detection
        self.layout_detection = layout_detection
        self.line_mode = mode
        self.pool = pool

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""

                QPushButton {
                    margin-top: 15px;
                    background-color: #ff0000;
                }

                QPushButton::hover {
                    color: #ffad00;
                }

            """)

        self.setCancelButton(self.cancel_btn)

        self.setStyleSheet("""

            background-color: #08081f;

            QProgressBar {
                background-color: #24272c;
                border-radius: 5px;
                border-width: 2px;
            }

            QProgressBar::chunk
            {
                background-color: #003d66;
                border-radius: 5px;
                margin: 3px 3px 3px 3px;
            }""")

        self.show()

    def exec(self):
        runner = OCRunner(self.data, self.line_detection, self.layout_detection, self.line_mode)
        runner.signals.sample.connect(self.handle_update_progress)
        runner.signals.error.connect(self.close)
        runner.signals.line_result.connect(self.handle_line_result)
        runner.signals.ocr_result.connect(self.handle_ocr_result)
        runner.signals.finished.connect(self.thread_complete)
        self.pool.start(runner)

    def handle_update_progress(self, value: int):
        print(f"Processing sample: {value}")

    def handle_error(self, error: str):
        print(f"Encountered Error: {error}")

    def handle_ocr_result(self, result: OCRResult):
        #self.sign_sam_result.emit(result)
        pass

    def handle_line_result(self, result: LineDataResult):
        #self.sign_batch_result.emit(result)
        pass

    def thread_complete(self):
        self.close()