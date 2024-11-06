import os
import cv2
from uuid import UUID
from typing import List, Tuple

import pyewts
from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QDialog, QLabel, QVBoxLayout, QHBoxLayout, \
    QProgressDialog, QPushButton, QListWidget, QListView, QListWidgetItem, QWidget, QTabWidget, QFormLayout, \
    QRadioButton, QProgressBar, QButtonGroup, QLineEdit, QComboBox

from BudaOCR.Data import BudaOCRData, OCResult, OCRModel, Theme, AppSettings, OCRSettings, \
    ExportFormat, Language, Encoding, OCRSample
from BudaOCR.Exporter import PageXMLExporter, JsonExporter, TextExporter
from BudaOCR.Inference import OCRPipeline
from BudaOCR.Runner import OCRBatchRunner, OCRunner
from BudaOCR.Utils import import_local_models


"""
Boiler plate to construct the Button groups based on the available settings
"""


# Languages
def build_languages(active_language: Language) -> Tuple[QButtonGroup, List[QRadioButton]]:
    buttons = []
    button_group = QButtonGroup()
    button_group.setExclusive(True)

    for lang in Language:
        button = QRadioButton(lang.name)
        button.setObjectName("OptionsRadio")
        buttons.append(button)

        if lang == active_language:
            button.setChecked(True)

        button_group.addButton(button)
        button_group.setId(button, lang.value)

    return button_group, buttons


# Export Formats
def build_exporter_settings(active_exporter: ExportFormat) -> Tuple[QButtonGroup, List[QRadioButton]]:
    exporter_buttons = []
    exporters_group = QButtonGroup()
    exporters_group.setExclusive(True)

    for exporter in ExportFormat:
        button = QRadioButton(exporter.name)
        button.setObjectName("OptionsRadio")
        exporter_buttons.append(button)

        if exporter == active_exporter:
            button.setChecked(True)

        exporters_group.addButton(button)
        exporters_group.setId(button, exporter.value)

    return exporters_group, exporter_buttons


# Encodigns
def build_encodings(active_encoding: Encoding) -> Tuple[QButtonGroup, List[QRadioButton]]:
    encoding_buttons = []
    encodings_group = QButtonGroup()
    encodings_group.setExclusive(True)

    for encoding in Encoding:
        button = QRadioButton(encoding.name)
        button.setObjectName("OptionsRadio")

        encoding_buttons.append(button)

        if encoding == active_encoding:
            button.setChecked(True)

        encodings_group.addButton(button)
        encodings_group.setId(button, encoding.value)

    return encodings_group, encoding_buttons


# Dewarping
def build_binary_selection(current_setting: bool) -> Tuple[QButtonGroup, List[QRadioButton]]:
    buttons = []
    button_group = QButtonGroup()
    button_group.setExclusive(True)

    yes_btn = QRadioButton("yes")
    no_btn = QRadioButton("no")
    yes_btn.setObjectName("OptionsRadio")
    no_btn.setObjectName("OptionsRadio")

    if current_setting:
        yes_btn.setChecked(True)
    else:
        no_btn.setChecked(True)

    button_group.addButton(yes_btn)
    button_group.addButton(no_btn)

    button_group.setId(no_btn, 0)
    button_group.setId(yes_btn, 1)

    buttons.append(yes_btn)
    buttons.append(no_btn)

    return button_group, buttons


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

        self.setText(message)

        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setStyleSheet("""
                               color: #ffffff;
                               font: bold 12px;
                               width: 240px;
                               height: 32px;
                               background-color: #A40021;
                               border-radius: 4px;

                               QPushButton::hover { 
                                   color: #ff0000;
                               }
                           """)

        self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)

        self.setStyleSheet("""
                    background-color: #1d1c1c;
                    color: #ffffff;
                    QPushButton {
                        width: 200px;
                        padding: 5px;
                        background-color: #A40021;
                    }
                """)


class ExportDialog(QDialog):
    def __init__(self, ocr_data: List[BudaOCRData], active_exporter: ExportFormat, active_encoding: Encoding):
        super().__init__()
        self.setObjectName("ExportDialog")
        self.ocr_data = ocr_data
        self.exporter = active_exporter
        self.encoding = active_encoding
        self.output_dir = "/"
        self.main_label = QLabel("Export OCR Data")
        self.main_label.setObjectName("OptionsLabel")
        self.exporter_group, self.exporter_buttons = build_exporter_settings(self.exporter)
        self.encodings_group, self.encoding_buttons = build_encodings(self.encoding)

        # build layout
        self.setWindowTitle("BudaOCR Export")
        self.setMinimumHeight(220)
        self.setMinimumWidth(600)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.export_dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setObjectName("")

        self.dir_select_btn = QPushButton("Select")
        self.dir_select_btn.setObjectName("SmallDialogButton")
        self.export_dir_layout.addWidget(self.dir_edit)
        self.export_dir_layout.addWidget(self.dir_select_btn)

        encoding_layout = QHBoxLayout()
        encoding_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for encoding in self.encoding_buttons:
            encoding_layout.addWidget(encoding)

        export_layout = QHBoxLayout()
        export_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in self.exporter_buttons:
            export_layout.addWidget(btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(self.ocr_data) - 1)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel", parent=self)
        self.cancel_btn.setObjectName("DialogButton")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)

        # bind signals
        self.ok_btn.clicked.connect(self.export)
        self.cancel_btn.clicked.connect(self.cancel)
        self.dir_select_btn.clicked.connect(self.select_export_dir)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.main_label)
        self.v_layout.addLayout(self.export_dir_layout)
        self.v_layout.addLayout(encoding_layout)
        self.v_layout.addLayout(export_layout)
        self.v_layout.addWidget(self.progress_bar)
        self.v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.v_layout)

        self.setStyleSheet(
            """
            background-color: #1d1c1c;
            color: #ffffff;

            QLabel {
                color: #000000;
            }
            """)

    def export(self):
        if os.path.isdir(self.output_dir):
            encoding_id = self.encodings_group.checkedId()
            exporters_id = self.exporter_group.checkedId()
            #converter = pyewts.pyewts()

            _encoding = Encoding(encoding_id)
            _exporter = ExportFormat(exporters_id)

            if _exporter == ExportFormat.XML:
                exporter = PageXMLExporter(self.output_dir)

                for idx, data in self.ocr_data.items():
                    img = cv2.imread(data.image_path)
                    exporter.export_lines(
                        img,
                        data.image_name,
                        data.lines,
                        data.ocr_text
                    )
                    self.progress_bar.setValue(idx)

            elif _exporter == ExportFormat.JSON:
                exporter = JsonExporter(self.output_dir)

                for idx, data in self.ocr_data.items():
                    img = cv2.imread(data.image_path)
                    lines = len(data.lines)

                    if len(lines) > 0:
                        exporter.export_lines(
                            img,
                            data.image_name,
                            data.lines,
                            data.ocr_text
                        )
                    self.progress_bar.setValue(idx)
            else:
                exporter = TextExporter(self.output_dir)

                for idx, data in self.ocr_data.items():
                    exporter.export_text(
                        data.image_name,
                        data.ocr_text
                    )
                    self.progress_bar.setValue(idx)

        else:
            dialog = NotificationDialog("Invalid Export Directory", "The selected output directory is not valid.")
            dialog.exec()

    def cancel(self):
        self.reject()

    def select_export_dir(self):
        dialog = ImportDirDialog()
        selected_dir = dialog.exec()

        if selected_dir == 1:
            _selected_dir = dialog.selectedFiles()[0]

            if os.path.isdir(_selected_dir):
                self.dir_edit.setText(_selected_dir)
                self.output_dir=_selected_dir
        else:
            note_dialog = NotificationDialog("Invalid Directory", "The selected directory is not valid.")
            note_dialog.exec()


class ModelListWidget(QWidget):
    def __init__(self, guid: UUID, title: str, encoder: str, architecture: str):
        super().__init__()
        self.guid = guid
        self.title = str(title)
        self.encoder = str(encoder)
        self.architecture = str(architecture)

        self.title_label = QLabel(self.title)
        self.encoder_label = QLabel(self.encoder)
        self.architecture_label = QLabel(self.architecture)
        self.download_btn = QPushButton('Download')
        self.delete_btn = QPushButton('Delete')

        # build layout
        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.title_label)
        self.h_layout.addWidget(self.encoder_label)
        self.h_layout.addWidget(self.architecture_label)
        self.h_layout.addWidget(self.download_btn)
        self.h_layout.addWidget(self.delete_btn)
        self.setLayout(self.h_layout)

        self.setStyleSheet("""
            color: #ffffff;
            width: 80%;
        """)


class ModelEntryWidget(QWidget):
    def __init__(self, guid: UUID, title: str, encoder: str, architecture: str):
        super().__init__()
        self.guid = guid
        self.title = str(title)
        self.encoder = str(encoder)
        self.architecture = str(architecture)

        self.title_label = QLabel(self.title)
        self.encoder_label = QLabel(self.encoder)
        self.architecture_label = QLabel(self.architecture)
        # build layout
        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.title_label)
        self.h_layout.addWidget(self.encoder_label)
        self.h_layout.addWidget(self.architecture_label)
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
        self.setObjectName("SettingsDialog")
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

        self.dark_theme_btn.setObjectName("OptionsRadio")
        self.light_theme_btn.setObjectName("OptionsRadio")

        self.theme_group = QButtonGroup()
        self.theme_group.setExclusive(True)
        self.theme_group.addButton(self.dark_theme_btn)
        self.theme_group.addButton(self.light_theme_btn)
        self.theme_group.setId(self.dark_theme_btn, Theme.Dark.value)
        self.theme_group.setId(self.light_theme_btn, Theme.Light.value)

        self.theme_group.setObjectName("OptionsRadio")

        if self.app_settings.theme == Theme.Dark:
            self.dark_theme_btn.setChecked(True)
            self.light_theme_btn.setChecked(False)
        else:
            self.dark_theme_btn.setChecked(False)
            self.light_theme_btn.setChecked(True)

        self.import_models_btn = QPushButton("Import Models")
        self.import_models_btn.clicked.connect(self.handle_model_import)

        self.exporter_group, self.exporter_buttons = build_exporter_settings(self.ocr_settings.exporter)
        self.encodings_group, self.encoding_buttons = build_encodings(self.app_settings.encoding)
        self.language_group, self.language_buttons = build_languages(self.app_settings.language)
        self.dewarp_group, self.dewarp_buttons = build_binary_selection(self.ocr_settings.dewarping)

        self.setWindowTitle("BudaOCR Settings")
        self.setMinimumHeight(400)
        self.setMinimumWidth(600)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # define layout
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setStyleSheet(
            """
                background-color: #3f3f3f;
                border: 0px;
        """)

        # General Settings Tab
        self.general_settings_tab = QWidget()
        self.general_settings_tab.setStyleSheet("""
            background-color: #172832;
            border: 0px;
        
        """)

        form_layout = QFormLayout()
        form_layout.setSpacing(20)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        ui_theme = QHBoxLayout()
        ui_theme.setAlignment(Qt.AlignmentFlag.AlignLeft)

        ui_theme.addWidget(self.dark_theme_btn)
        ui_theme.addWidget(self.light_theme_btn)

        language_layout = QHBoxLayout()
        language_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in self.language_buttons:
            language_layout.addWidget(btn)

        theme_label = QLabel("UI Theme")
        theme_label.setObjectName("OptionsLabel")
        theme_label.setFixedWidth(160)

        language_label = QLabel("Language")
        language_label.setObjectName("OptionsLabel")
        language_label.setFixedWidth(160)

        form_layout.addRow(theme_label, ui_theme)
        form_layout.addRow(language_label, language_layout)
        self.general_settings_tab.setLayout(form_layout)

        # OCR Models Tab
        self.ocr_label = QLabel("Available OCR Models")
        self.ocr_label.setStyleSheet("""
                color: #ffffff
        """)

        self.ocr_models_tab = QWidget()
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.ocr_label)
        h_layout.addWidget(self.import_models_btn)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.model_list)
        self.ocr_models_tab.setLayout(v_layout)

        # OCR Settings Tab
        self.ocr_settings_tab = QWidget()
        self.ocr_settings_tab.setContentsMargins(0, 0, 0, 0)
        self.ocr_settings_tab.setStyleSheet("""
                    background-color: #172832;
          
                """)

        form_layout = QFormLayout()
        form_layout.setSpacing(20)

        form_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        encoding_layout = QHBoxLayout()
        encoding_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for encoding in self.encoding_buttons:
            encoding_layout.addWidget(encoding)

        dewarping_layout = QHBoxLayout()
        dewarping_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in self.dewarp_buttons:
            dewarping_layout.addWidget(btn)

        export_layout = QHBoxLayout()
        export_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for btn in self.exporter_buttons:
            export_layout.addWidget(btn)

        encoding_label = QLabel("Encoding")
        dewarping_label = QLabel("Dewarping")
        export_label = QLabel("Export Formats")

        encoding_label.setObjectName("OptionsLabel")
        dewarping_label.setObjectName("OptionsLabel")
        export_label.setObjectName("OptionsLabel")

        encoding_label.setFixedWidth(160)
        dewarping_label.setFixedWidth(160)
        export_label.setFixedWidth(160)

        form_layout.addRow(encoding_label, encoding_layout)
        form_layout.addRow(dewarping_label, dewarping_layout)
        form_layout.addRow(export_label, export_layout)

        self.ocr_settings_tab.setLayout(form_layout)

        # build entire Layout
        self.settings_tabs.addTab(self.general_settings_tab, "General")
        self.settings_tabs.addTab(self.ocr_models_tab, "OCR Models")
        self.settings_tabs.addTab(self.ocr_settings_tab, "OCR Settings")

        self.main_v_layout = QVBoxLayout()
        self.main_v_layout.addWidget(self.settings_tabs)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel", parent=self)
        self.cancel_btn.setObjectName("DialogButton")

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

        self.setStyleSheet(
            """
            background-color: #1d1c1c;
            color: #ffffff;
            
            QFormLayout {
                background-color: #1d1c1c;  
            }
        
            QLabel {
                color: #ffffff;
            }
            QDialogButtonBox::Ok {
                height: 32px;
                width: 64px;
            }
            QDialogButtonBox::Cancel {
                height: 32px;
                width: 64px;
            }
            
            QTabBar {
                background-color: #ff0000;
            }
                        
            QTabWidget::QTabBar {
                background-color: #ff0000;  
            }
            
            QTabBar::tab
            {
                 background-color: #3f3f3f;
            }
            
             QTabWidget::pane {
                    border: 0px;
                    background-color: #0000ff;
                    padding-top: 20px;
                }
                
            QTabWidget::tab-bar {
                background-color: #ff0000;
                color: #0000ff;
                left: 20px;
                height: 36px;
                border-radius: 4px;
                alignment: center; 
            }
                
            """)

        self.build_model_overview()

    def handle_accept(self):
        self.accept()

    def handle_reject(self):
        self.reject()

    def build_model_overview(self):
        self.model_list.clear()

        for model in self.ocr_models:
            model_item = QListWidgetItem(self.model_list)
            #model_widget = ModelListWidget(guid=uuid.uuid1(),title=model.name)
            model_widget = ModelEntryWidget(
                guid=model.guid,
                title=model.name,
                encoder=model.config.encoder.name,
                architecture=model.config.architecture.name
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

        # fetch settings
        theme_id = self.theme_group.checkedId()
        self.app_settings.theme = Theme(theme_id)

        language_id = self.language_group.checkedId()
        self.app_settings.language = Language(language_id)

        encoding_id = self.encodings_group.checkedId()
        self.app_settings.encoding = Encoding(encoding_id)

        exporters_id = self.exporter_group.checkedId()
        self.ocr_settings.exporter = ExportFormat(exporters_id)

        dewarp_id = self.dewarp_group.checkedId()
        do_dewarp = bool(dewarp_id)
        self.ocr_settings.dewarping = do_dewarp

        return self.app_settings, self.ocr_settings


class BatchOCRDialog(QDialog):
    sign_ocr_result = Signal(OCResult)

    def __init__(self, data: List[BudaOCRData], ocr_pipeline: OCRPipeline, ocr_models: List[OCRModel], ocr_settings: OCRSettings, threadpool: QThreadPool):
        super().__init__()
        self.setObjectName("BatchOCRDialog")
        self.data = data
        self.pipeline = ocr_pipeline
        self.ocr_models = ocr_models
        self.ocr_settings = ocr_settings
        self.threadpool = threadpool
        self.runner = None
        self.output_dir = ""

        self.setMinimumWidth(600)
        self.setMaximumWidth(1200)
        self.setFixedHeight(340)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("DialogProgressBar")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(self.data)-1)

        self.start_process_btn = QPushButton("Start")
        self.start_process_btn.setObjectName("SmallDialogButton")
        self.cancel_process_btn = QPushButton("Cancel")
        self.cancel_process_btn.setObjectName("SmallDialogButton")

        # settings elements
        # Exports
        self.exporter_group, self.exporter_buttons = build_exporter_settings(self.ocr_settings.exporter)
        self.encodings_group, self.encoding_buttons = build_encodings(self.ocr_settings.output_encoding)
        self.dewarp_group, self.dewarp_buttons = build_binary_selection(self.ocr_settings.dewarping)

        # build layout
        self.progress_layout = QHBoxLayout()
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.start_process_btn)
        self.progress_layout.addWidget(self.cancel_process_btn)

        self.button_h_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setObjectName("DialogButton")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("DialogButton")

        self.button_h_layout.addWidget(self.ok_btn)
        self.button_h_layout.addWidget(self.cancel_btn)

        self.v_layout = QVBoxLayout()
        self.label = QLabel("Batch Processing")
        self.label.setObjectName("OptionsLabel")
        self.label.setStyleSheet("""
            font-weight: bold;
        """)

        self.export_dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setObjectName("DialogLineEdit")

        self.dir_select_btn = QPushButton("select")
        self.dir_select_btn.setObjectName("SmallDialogButton")
        self.export_dir_layout.addWidget(self.dir_edit)
        self.export_dir_layout.addWidget(self.dir_select_btn)

        self.form_layout = QFormLayout()
        self.form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        self.model_selection = QComboBox()
        self.model_selection.setStyleSheet("""
                color: #ffffff;
                background: #434343;
                border: 2px solid #ced4da;
                border-radius: 4px;
            """)

        if self.ocr_models is not None and len(self.ocr_models) > 0:
            for model in self.ocr_models:
                self.model_selection.addItem(model.name)

        self.model_selection.currentIndexChanged.connect(self.on_select_ocr_model)

        encoding_layout = QHBoxLayout()
        for btn in self.encoding_buttons:
            encoding_layout.addWidget(btn)

        dewarping_layout = QHBoxLayout()
        for btn in self.dewarp_buttons:
            dewarping_layout.addWidget(btn)

        export_layout = QHBoxLayout()
        for btn in self.exporter_buttons:
            export_layout.addWidget(btn)

        encoding_label = QLabel("Encoding")
        encoding_label.setObjectName("OptionsLabel")

        dewarping_label = QLabel("Dewarping")
        dewarping_label.setObjectName("OptionsLabel")

        export_labels = QLabel("Export Formats")
        export_labels.setObjectName("OptionsLabel")

        self.form_layout.addRow(encoding_label, encoding_layout)
        self.form_layout.addRow(dewarping_label, dewarping_layout)
        self.form_layout.addRow(export_labels, export_layout)

        self.status_layout = QHBoxLayout()
        self.status_label = QLabel("Status")
        self.status_label.setObjectName("OptionsLabel")
        self.status = QLabel("")
        self.status.setObjectName("OptionsLabel")

        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.status)

        self.v_layout.addWidget(self.label)
        self.v_layout.addWidget(self.model_selection)
        self.v_layout.addLayout(self.export_dir_layout)
        self.v_layout.addLayout(self.form_layout)
        self.v_layout.addLayout(self.progress_layout)
        self.v_layout.addLayout(self.status_layout)
        self.v_layout.addLayout(self.button_h_layout)
        self.setLayout(self.v_layout)

        # bind signals
        self.dir_select_btn.clicked.connect(self.select_export_dir)
        self.start_process_btn.clicked.connect(self.start_process)
        self.cancel_process_btn.clicked.connect(self.cancel_process)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.setStyleSheet("""
            background-color: #1d1c1c;
            color: #ffffff;
            
            QLineEdit {
                color: #ffffff;
                background-color: #474747;
                border: 2px solid #343942;
                border-radius: 8px;
                padding: 6px;
                text-align: left;
            }
        
        """)

    def select_export_dir(self):
        dialog = ImportDirDialog()
        selected_dir = dialog.exec()

        if selected_dir == 1:
            _selected_dir = dialog.selectedFiles()[0]

            if os.path.isdir(_selected_dir):
                self.dir_edit.setText(_selected_dir)
                self.output_dir=_selected_dir
        else:
            note_dialog = NotificationDialog("Invalid Directory", "The selected directory is not valid.")
            note_dialog.exec()

    def on_select_ocr_model(self, index: int):
        self.pipeline.update_ocr_model(self.ocr_models[index].config)

    def start_process(self):

        if os.path.isdir(self.output_dir):
            encoding_id = self.encodings_group.checkedId()
            encoding = Encoding(encoding_id)

            self.runner = OCRBatchRunner(self.data, self.pipeline, output_encoding=encoding)
            self.runner.signals.sample.connect(self.handle_update_progress)
            self.runner.signals.finished.connect(self.finish)
            self.threadpool.start(self.runner)
            self.status.setText("Running")

        else:
            note_dialog = NotificationDialog("No Ouput Directory", "Please select an output directory.")
            note_dialog.exec()

    def handle_update_progress(self, sample: OCRSample):
        self.progress_bar.setValue(sample.cnt)
        file_name = self.data[sample.cnt].image_name
        out_file = os.path.join(self.output_dir, f"{file_name}.txt")

        with open(out_file, "w", encoding="utf-8") as f:
            for line in sample.result.text:
                f.write(f"{line}\n")

        self.sign_ocr_result.emit(sample.result)

    def finish(self):
        print(f"Thread Completed")
        self.runner = None
        self.status.setText("Finished")

    def cancel_process(self):
        if self.runner is not None:
            self.runner.stop = True

class OCRDialog(QProgressDialog):
    sign_ocr_result = Signal(OCResult)

    def __init__(self, pipeline: OCRPipeline, settings: OCRSettings, data: BudaOCRData, pool: QThreadPool):
        super(OCRDialog, self).__init__()
        self.setObjectName("OCRDialog")
        self.setMinimumWidth(500)
        self.setWindowTitle("OCR Progress")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimum(0)
        self.setMaximum(0)
        self.pipeline = pipeline
        self.settings = settings
        self.data = data
        self.pool = pool
        self.runner = None
        self.result = None

        # build layout
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
            }
        """)

        self.show()

    def exec(self):
        print(f"Running Async OCR")
        runner = OCRunner(self.data, self.pipeline, self.settings)
        runner.signals.error.connect(self.handle_error)
        runner.signals.ocr_result.connect(self.handle_ocr_result)
        runner.signals.finished.connect(self.thread_complete)
        self.pool.start(runner)

    def handle_error(self, error: str):
        print(f"Encountered Error: {error}")

    def handle_ocr_result(self, result: OCResult):
        print(f"Handling ocr result: {result}")
        self.sign_ocr_result.emit(result)

    def thread_complete(self):
        print(f"Thread Complete")
        #self.close()