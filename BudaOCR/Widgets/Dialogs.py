import os
import uuid
from uuid import UUID
from typing import List
from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QDialog, QLabel, QVBoxLayout, QHBoxLayout, \
    QProgressDialog, QPushButton, QListWidget, QListView, QListWidgetItem, QWidget, QTabWidget, QFormLayout, \
    QRadioButton, QCheckBox

from BudaOCR.Data import BudaOCRData, OCResult, LineDataResult, OCRModel, Theme, AppSettings, OCRSettings, \
    ExportFormat
from BudaOCR.Inference import LayoutDetection, LineDetection
from BudaOCR.Runner import OCRunner
from BudaOCR.Utils import import_local_models


class ImportFilesDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ImportFilesDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.setNameFilter("Images (*.png *.jpg *.tif *.tiff)")
        self.setViewMode(QFileDialog.ViewMode.List)


class ImportDirDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ImportDirDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.Directory)


class ConfirmationDialog(QMessageBox):
    def __init__(self, title: str, message: str, show_cancel: bool = True):
        super().__init__()
        self.setObjectName("ConfirmWindow")
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self.setMinimumHeight(440)
        self.setIcon(QMessageBox.Icon.Information)
        self.setText(message)

        self.ok_btn = QPushButton("Ok")
        self.cancel_btn = QPushButton("Cancel")

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.ok_btn.setStyleSheet("""
                color: #000000;
                font: bold 12px;
                width: 240px;
                height: 32px;
                background-color: #ffad00;
                border: 2px solid #ffad00;
                border-radius: 4px;

                QPushButton::hover { 
                    color: #ff0000;
                }

            """)

        self.cancel_btn.setStyleSheet("""
                color: #000000;
                font: bold 12px; 
                width: 240px;
                height: 32px;
                background-color: #ffad00;
                border: 2px solid #ffad00;
                border-radius: 4px;

                QPushButton::hover {
                    color: #ff0000;
                }
            """)

        if show_cancel:
            self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)
            self.addButton(self.cancel_btn, QMessageBox.ButtonRole.NoRole)
        else:
            self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)

        self.setStyleSheet("""
                background-color: #292951;
                color: #ffffff;
        """)


class NotificationDialog(QMessageBox):
    def __init__(self, title: str, message: str):
        super().__init__()
        self.setObjectName("NotificationWindow")
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self.setMinimumHeight(440)
        self.setIcon(QMessageBox.Icon.Information)
        self.setStandardButtons(QMessageBox.Ok)
        self.setText(message)

        self.setStyleSheet("""
                    color: #ffffff;
                    QPushButton {
                        width: 200px;
                        padding: 5px;
                        background-color: #4d4d4d;
                    }
                """)


class ModelListWidget(QWidget):
    def __init__(self, guid: UUID, title: str):
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
            width: 80%;
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

        self.setStyleSheet(
            """
            background-color: #172832;
            border-radius: 4px;

            """)

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(
            item
        )  # returns an instance of CanvasHierarchyEntry

        if isinstance(_list_item_widget, ModelListWidget):
            print(f"Clicked on Model: {_list_item_widget.title}")
            self.sign_on_selected_item.emit(_list_item_widget.guid)


class SettingsDialog(QDialog):
    def __init__(self, app_settings: AppSettings, ocr_settings: OCRSettings, ocr_models: List[OCRModel]):
        super().__init__()
        self.app_settings = app_settings
        self.ocr_settings = ocr_settings
        self.ocr_models = ocr_models
        self.model_list = ModelList(self)

        self.selected_theme = Theme.Dark
        self.selected_exporters = []

        # Settings
        # Theme
        self.dark_theme_btn = QRadioButton("Dark")
        self.light_theme_btn = QRadioButton("Light")

        if self.app_settings.theme == Theme.Dark:
            self.dark_theme_btn.setChecked(True)
            self.light_theme_btn.setChecked(False)
        else:
            self.dark_theme_btn.setChecked(False)
            self.light_theme_btn.setChecked(True)

        self.import_models_btn = QPushButton("Import Models")
        self.import_models_btn.clicked.connect(self.handle_model_import)

        # Exports
        self.export_xml = QCheckBox("XML")
        self.export_json = QCheckBox("Json")
        self.export_text = QCheckBox("Text")

        for export_format in self.app_settings.export_formats:
            if export_format == ExportFormat.XML:
                self.export_xml.setChecked(True)
            if export_format == ExportFormat.JSON:
                self.export_json.setChecked(True)
            if export_format == ExportFormat.Text:
                self.export_text.setChecked(True)

        self.setWindowTitle("BudaOCR Settings")
        self.setMinimumHeight(400)
        self.setMinimumWidth(600)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # define layout
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setContentsMargins(0, 0, 0, 0)

        self.settings_tabs.setStyleSheet(
            """
                QTabWidget::pane {
                    border: None;
                    padding-top: 20px;
                }
        """)

        self.general_settings_tab = QWidget()
        layout = QFormLayout()
        ui_theme = QHBoxLayout()
        ui_theme.addWidget(self.dark_theme_btn)
        ui_theme.addWidget(self.light_theme_btn)

        language = QHBoxLayout()
        language.addWidget(QRadioButton("English"))
        language.addWidget(QRadioButton("German"))
        language.addWidget(QRadioButton("French"))
        language.addWidget(QRadioButton("Tibetan"))
        language.addWidget(QRadioButton("Chinese"))

        layout.addRow(QLabel("UI Theme"), ui_theme)
        layout.addRow(QLabel("Language"), language)
        self.general_settings_tab.setLayout(layout)

        self.ocr_models_tab = QWidget()
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Available OCR Models"))
        h_layout.addWidget(self.import_models_btn)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.model_list)
        self.ocr_models_tab.setLayout(v_layout)

        self.ocr_settings_tab = QWidget()
        form_layout = QFormLayout()
        encoding_layout = QHBoxLayout()
        encoding_layout.addWidget(QRadioButton("Unicode"))
        encoding_layout.addWidget(QRadioButton("Wylie"))

        dewarping_layout = QHBoxLayout()
        dewarping_layout.addWidget(QRadioButton("yes"))
        dewarping_layout.addWidget(QRadioButton("no"))

        export_layout = QHBoxLayout()
        export_layout.addWidget(self.export_xml)
        export_layout.addWidget(self.export_json)
        export_layout.addWidget(self.export_text)

        form_layout.addRow(QLabel("Encoding"), encoding_layout)
        form_layout.addRow(QLabel("Dewarping"), dewarping_layout)
        form_layout.addRow(QLabel("Export Formats"), export_layout)

        self.ocr_settings_tab.setLayout(form_layout)

        self.settings_tabs.addTab(self.general_settings_tab, "General")
        self.settings_tabs.addTab(self.ocr_models_tab, "OCR Models")
        self.settings_tabs.addTab(self.ocr_settings_tab, "OCR Settings")

        self.main_v_layout = QVBoxLayout()
        self.main_v_layout.addWidget(self.settings_tabs)

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

        self.import_models_btn.setStyleSheet("""
            QPushButton {
                    color: #A40021;
                    background-color: #fce08d;
                    border-radius: 4px;
                    height: 18;
                }
                
            QPushButton::hover {
                    color: #ffad00;
                }
                
        """)
        self.ok_btn.setStyleSheet(
            """
                QPushButton {
                    margin-top: 15px;
                    background-color: #A40021;
                    border-radius: 4px;
                    height: 24;
                }

                QPushButton::hover {
                    color: #ffad00;
                }
            """)

        self.cancel_btn.setStyleSheet(
            """
                QPushButton {
                    margin-top: 15px;
                    background-color: #A40021;
                    border-radius: 4px;
                    height: 24;
                }

                QPushButton::hover {
                    color: #ffad00;
                }
            """)

        self.setStyleSheet(
            """
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

        self.build_model_overview()

    def handle_accept(self):
        self.accept()

    def handle_reject(self):
        self.reject()

    def build_model_overview(self):
        self.model_list.clear()
        print(f"Building model overview...")

        for model in self.ocr_models:
            print(f"Model: {model.name}")
            model_item = QListWidgetItem(self.model_list)
            model_widget = ModelListWidget(
                guid=uuid.uuid1(),
                title=model.name
            )

            model_item.setSizeHint(model_widget.sizeHint())
            self.model_list.addItem(model_item)
            self.model_list.setItemWidget(model_item, model_widget)

    def clear_models(self):
        self.model_list.clear()

    def handle_model_import(self):
        _dialog = ImportDirDialog()
        selected_dir = _dialog.exec()

        if selected_dir == 1:
            _selected_dir = _dialog.selectedFiles()[0]

            if os.path.isdir(_selected_dir):
                try:
                    imported_models = import_local_models(_selected_dir)
                    confirm_dialog = ConfirmationDialog(
                        title="Confirm Model Import",
                        message="Do you want to import the new models and replace the old ones?"
                    )
                    confirm_dialog.exec()
                    result = confirm_dialog.result()

                    if result == 2:
                        print(f"Result: {result}")
                        self.ocr_models = imported_models
                        self.build_model_overview()
                    else:
                        print("Skipping import of new models")

                except BaseException as e:
                    error_dialog = NotificationDialog("Model import failed", f"Importing Models Failed: {e}")
                    error_dialog.exec()

    def exec(self):
        super().exec()
        return self.app_settings, self.ocr_settings


class OCRBatchProgress(QProgressDialog):
    sign_line_result = Signal(LineDataResult)
    sign_ocr_result = Signal(OCResult)

    def __init__(self, data: list[BudaOCRData], pool: QThreadPool, parent=None):
        super(OCRBatchProgress, self).__init__(parent)
        self.setObjectName("OCRDialog")
        self.setMinimumWidth(500)
        self.setWindowTitle("OCR Progress")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimum(0)
        self.setMaximum(0)

        self.data = data
        self.pool = pool

        self.start_btn = QPushButton("Start")
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
        """
        runner = OCRunner(self.data, self.line_detection, self.layout_detection, self.line_mode)
        runner.signals.sample.connect(self.handle_update_progress)
        runner.signals.error.connect(self.close)
        runner.signals.line_result.connect(self.handle_line_result)
        runner.signals.ocr_result.connect(self.handle_ocr_result)
        runner.signals.finished.connect(self.thread_complete)
        self.pool.start(runner)
        """

    def handle_update_progress(self, value: int):
        print(f"Processing sample: {value}")

    def handle_error(self, error: str):
        print(f"Encountered Error: {error}")

    def handle_ocr_result(self, result: OCResult):
        #self.sign_sam_result.emit(result)
        pass

    def handle_line_result(self, result: LineDataResult):
        #self.sign_batch_result.emit(result)
        pass

    def thread_complete(self):
        self.close()