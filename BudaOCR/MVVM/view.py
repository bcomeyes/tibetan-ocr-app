import os
from uuid import UUID

import cv2
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QSplitter
from PySide6.QtCore import Signal, Qt, QThreadPool
from typing import Dict
from BudaOCR.Config import save_app_settings, save_ocr_settings, LINES_CONFIG, LAYOUT_CONFIG
from BudaOCR.Data import OpStatus, Platform, BudaOCRData, OCRModel, OCResult
from BudaOCR.Inference import OCRPipeline
from BudaOCR.Utils import get_filename, generate_guid, read_line_model_config, read_layout_model_config
from BudaOCR.Widgets.Dialogs import NotificationDialog, SettingsDialog, BatchOCRDialog, OCRDialog, ExportDialog
from BudaOCR.Widgets.Layout import HeaderTools, ImageGallery, Canvas, TextView
from BudaOCR.MVVM.viewmodel import BudaDataViewModel, BudaSettingsViewModel
from BudaOCR.IO import TextExporter
from BudaOCR.Styles import DARK


class MainView(QWidget):
    sign_handle_import = Signal(list)
    sign_handle_page_select = Signal(int)
    sign_on_file_save = Signal()
    sign_run_ocr = Signal(UUID)
    sign_run_batch_ocr = Signal()
    sign_handle_settings = Signal()

    def __init__(self, data_view: BudaDataViewModel, settings_view: BudaSettingsViewModel):
        super().__init__()
        self.setObjectName("MainView")
        self._data_view = data_view
        self._settings_view = settings_view

        self.header_tools = HeaderTools(self._data_view, self._settings_view)
        self.canvas = Canvas()
        self.text_view = TextView()
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        #self.splitter.setCollapsible(0, False)

        # build layout
        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.header_tools)

        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.text_view)
        self.v_layout.addWidget(self.splitter)

        self.setLayout(self.v_layout)

        # connect to view model signals
        self._data_view.dataSelected.connect(self.set_data)
        self._data_view.dataChanged.connect(self.update_data)
        self._data_view.recordChanged.connect(self.set_data)

        # connect to tool signals
        self.header_tools.toolbox.sign_new.connect(self.handle_new)
        self.header_tools.toolbox.sign_import_files.connect(self.handle_import)
        self.header_tools.toolbox.sign_save.connect(self.handle_file_save)
        self.header_tools.toolbox.sign_on_select_model.connect(self.handle_model_selection)
        self.header_tools.toolbox.sign_run.connect(self.handle_run)
        self.header_tools.toolbox.sign_run_all.connect(self.handle_batch_run)
        self.header_tools.toolbox.sign_settings.connect(self.handle_settings)
        self.header_tools.page_switcher.sign_on_page_changed.connect(self.handle_update_page)
        self.current_guid = None


    def handle_new(self):
        self._data_view.clear_data()

    def update_data(self, data: list[BudaOCRData]):
        self.header_tools.update_page_count(len(data))

    def handle_import(self, files: list[str]):
        self.sign_handle_import.emit(files)

    def handle_file_save(self):
        self.sign_on_file_save.emit()

    def handle_model_selection(self, ocr_model: OCRModel):
        self._settings_view.select_ocr_model(ocr_model)


    def handle_run(self):
        if self.current_guid is not None:
            self.sign_run_ocr.emit(self.current_guid)
        else:
            dialog = NotificationDialog("No data", "Project contains no data.")
            dialog.exec()

    def handle_batch_run(self):
        self.sign_run_batch_ocr.emit()

    def handle_settings(self):
        self.sign_handle_settings.emit()

    def handle_update_page(self, index: int):
        self._data_view.select_data_by_index(index)

    def set_data(self, data: BudaOCRData):
        self.canvas.set_preview(data)
        self.text_view.update_text(data.ocr_text)
        self.current_guid = data.guid


class AppView(QWidget):
    def __init__(self,
                 dataview_model: BudaDataViewModel,
                 settingsview_model: BudaSettingsViewModel,
                 platform: Platform,
                 max_width: int,
                 max_height: int):
        super().__init__()
        self.setObjectName("MainWindow")
        self.platform = platform
        self.threadpool = QThreadPool()
        self._dataview_model = dataview_model
        self._settingsview_model = settingsview_model

        self.image_gallery = ImageGallery(self._dataview_model)
        self.main_container = MainView(self._dataview_model, self._settingsview_model)
        self.main_container.setContentsMargins(0, 0, 0, 0)
        # build layout
        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumWidth(int(max_width * 0.5))
        self.setMaximumWidth(max_width)
        self.setMinimumHeight(int(max_height * 0.5))
        self.setMaximumHeight(max_height)

        self.main_layout = QHBoxLayout()
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_layout.addWidget(self.image_gallery)
        self.main_layout.addWidget(self.main_container)

        # self.main_layout.addWidget(self.footer)
        self.setLayout(self.main_layout)

        # connect view model signals
        self._settingsview_model.ocrModelChanged.connect(self.update_ocr_model)

        # connect layout signals
        self.main_container.sign_handle_import.connect(self.handle_file_import)
        self.main_container.sign_handle_page_select.connect(self.select_page)
        self.main_container.sign_on_file_save.connect(self.save)
        self.main_container.sign_run_ocr.connect(self.run_ocr)
        #self.main_container.sign_run_ocr.connect(self.run_ocr_async)
        self.main_container.sign_run_batch_ocr.connect(self.run_batch_ocr)
        self.main_container.sign_handle_settings.connect(self.handle_settings)

        self.text_exporter = TextExporter()
        self.app_images = {}
        self.current_image = None

        # inference sessions
        self.line_model_config = read_line_model_config(LINES_CONFIG)
        self.layout_model_config = read_layout_model_config(LAYOUT_CONFIG)
        _ocr_model = self._settingsview_model.get_current_ocr_model()

        print(f"Creating Default OCRPipeline: {_ocr_model.name}")
        self.ocr_pipeline = OCRPipeline(self.platform, _ocr_model.config, self.layout_model_config)

        self.setStyleSheet("""
            QFrame::TextView {
                color: #ffffff;
                background-color: #100F0F;
                border: 2px solid #100F0F; 
                border-radius: 8px;
            }
        """)

        self.show()

    def handle_file_import(self, files: list[str]):
        new_data = {}

        for idx, file_path in enumerate(files):
            if os.path.isfile(file_path):
                file_name = get_filename(file_path)
                guid = generate_guid(idx)
                palmtree_data = BudaOCRData(
                    guid=guid,
                    image_path=file_path,
                    image_name=file_name,
                    ocr_text=[],
                    lines=None,
                    preview=None
                )
                new_data[guid] = palmtree_data

        self._dataview_model.add_data(new_data)

    def save(self):
        _ocr_data = self._dataview_model.get_data()
        ocred_lines = 0

        for k, data in _ocr_data.items():
            if data.ocr_text is not None and len(data.ocr_text) > 0:
                ocred_lines += 1

        if not len(_ocr_data) > 0:
            info_box = NotificationDialog(
                "Saving Annotations",
                "The current project contains no data."
            )
            info_box.exec()

        elif not ocred_lines > 0:
            info_box = NotificationDialog(
                "Saving Annotations",
                "The current contains no OCR data. Please run OCR first."
            )
            info_box.exec()

        else:
            _ocr_settings = self._settingsview_model.get_ocr_settings()
            dialog = ExportDialog(_ocr_data, _ocr_settings.exporter, _ocr_settings.output_encoding)
            dialog.setStyleSheet(DARK)
            dialog.exec()

    def select_page(self, index: int):
        self.image_gallery.select_page(index)

    def run_ocr(self, guid: UUID):
        data = self._dataview_model.get_data_by_guid(guid)

        if os.path.isfile(data.image_path):

            img = cv2.imread(data.image_path)
            status, ocr_result = self.ocr_pipeline.run_ocr(img)

            if status == OpStatus.SUCCESS:
                mask, line_data, page_text = ocr_result
                self._dataview_model.update_ocr_data(guid, page_text)
                self._dataview_model.update_page_data(guid, line_data, mask)
            else:
                dialog = NotificationDialog("Failed Running OCR", "Failed to run OCR on selected image.")
                dialog.exec()
        else:
            dialog = NotificationDialog("Image not found", "The selected image could not be read from disk.")
            dialog.exec()

    def run_ocr_async(self, guid: UUID):
        _data = self._dataview_model.get_data_by_guid(guid)
        _settings = self._settingsview_model.get_ocr_settings()
        ocr_dialog = OCRDialog(self.ocr_pipeline, _settings, _data, self.threadpool)
        ocr_dialog.sign_ocr_result.connect(self.update_ocr_result)
        ocr_dialog.exec()

    def run_batch_ocr(self):
        _data = self._dataview_model.get_data()
        _data = list(_data.values())

        if _data is not None and len(_data) > 0:
            batch_dialog = BatchOCRDialog(
                data=_data,
                ocr_pipeline=self.ocr_pipeline,
                ocr_models=self._settingsview_model.get_ocr_models(),
                ocr_settings=self._settingsview_model.get_ocr_settings(),
                threadpool=self.threadpool
            )
            batch_dialog.sign_ocr_result.connect(self.update_ocr_result)

            batch_dialog.setStyleSheet(DARK)
            batch_dialog.exec()
        else:
            dialog = NotificationDialog("No data", "Project contains no data.")
            dialog.exec()

    def update_ocr_result(self, result: OCResult, silent: bool = False):
        if result is not None:
            self._dataview_model.update_ocr_data(result.guid, result.text, silent)
            self._dataview_model.update_page_data(result.guid, result.lines, result.mask, silent)

        else:
            dialog = NotificationDialog("Failed Running OCR", "Failed to run OCR on selected image.")
            dialog.exec()

    def handle_ocr_batch_result(self, batch_result: Dict[UUID, OCResult]):
        if batch_result is not None:
            for _, result in batch_result.values():
                self.update_ocr_result(result, silent=True)
        else:
            dialog = NotificationDialog("Failed Running OCR", "Failed to run OCR on selected image.")
            dialog.exec()

    def handle_settings(self):
        _models = self._settingsview_model.get_ocr_models()
        _app_settings = self._settingsview_model.get_app_settings()
        _ocr_settings = self._settingsview_model.get_ocr_settings()

        dialog = SettingsDialog(_app_settings, _ocr_settings, _models)
        dialog.setStyleSheet(DARK)

        app_settings, ocr_settings = dialog.exec()

        # TODO: add ovewrite confirmation or so ?
        save_app_settings(app_settings)
        save_ocr_settings(ocr_settings)

    def update_ocr_model(self, ocr_model: OCRModel):
        print(f"Updating OCR Pipeline with OCR model: {ocr_model.name}")
        self.ocr_pipeline.update_ocr_model(ocr_model.config)