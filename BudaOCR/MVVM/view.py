import os
from uuid import UUID

import cv2
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QSplitter
from PySide6.QtCore import Signal, Qt, QThreadPool
from huggingface_hub import HfApi

from BudaOCR.Data import BudaOCRData, OCRSettings
from BudaOCR.Utils import get_filename, generate_guid, read_line_model_config, read_layout_model_config, get_line_data, \
    extract_line_images
from BudaOCR.Inference import LineDetection, LayoutDetection, OCRInference
from BudaOCR.Widgets.Dialogs import NotificationDialog, SettingsWindow
from BudaOCR.Widgets.Layout import HeaderTools, OCRTools, ImageGallery, Canvas, TextView


from BudaOCR.MVVM.viewmodel import BudaViewModel
from BudaOCR.Controller import ImageController
from BudaOCR.IO import TextExporter

LINE_MODEL_PATH = "Models/Lines/PhotiLines/config.json"
LAYOUT_MODEL_PATH = "Models/Layout/2024-6-6_8-58/config.json"
OCR_MODEL_PATH = "Models/OCR/Woodblock/config.json"


class MainView(QWidget):
    sign_handle_import = Signal(list)
    sign_handle_page_select = Signal(int)
    sign_on_file_save = Signal()
    sign_run_ocr = Signal(UUID)

    def __init__(self, view_model: BudaViewModel, controller: ImageController):
        super().__init__()
        self.hf_api = HfApi()
        self._view_model = view_model
        self.controller = controller

        self.header_toos = HeaderTools()
        self.canvas = Canvas()
        self.text_view = TextView()

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        #self.splitter.setCollapsible(1, False)

        # build layout
        self.v_layout = QVBoxLayout()
        self.v_layout.setSpacing(20)
        self.v_layout.addWidget(self.header_toos)

        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.text_view)

        self.v_layout.addWidget(self.splitter)
        self.setLayout(self.v_layout)

        # connect to view model signals
        self._view_model.dataSelected.connect(self.set_data)
        self._view_model.dataChanged.connect(self.update_data)
        self._view_model.recordChanged.connect(self.set_data)

        # connect to tool signals
        self.header_toos.toolbox.sign_new.connect(self.handle_new)
        self.header_toos.toolbox.sign_import_files.connect(self.handle_import)
        self.header_toos.toolbox.sign_save.connect(self.handle_file_save)
        self.header_toos.toolbox.sign_run.connect(self.handle_run)
        self.header_toos.toolbox.sign_settings.connect(self.handle_settings)
        self.header_toos.page_switcher.sign_on_page_changed.connect(self.handle_update_page)

        # connect to controller signals
        self.controller.on_select_image.connect(self.selecting_image)
        self.current_guid = None


    def handle_new(self):
        print("handling new signal")
        self._view_model.clear_data()

    def update_data(self, data: list[BudaOCRData]):
        self.header_toos.update_page_count(len(data))

    def handle_import(self, files: list[str]):
        print(f"Handling import: {files}")
        self.sign_handle_import.emit(files)

    def handle_file_save(self):
        print("Handling FileSave")
        self.sign_on_file_save.emit()

    def handle_run(self):
        print(f"Running OCR....")
        if self.current_guid is not None:
            self.sign_run_ocr.emit(self.current_guid)

    def handle_settings(self):
        print("Handling Settings")
        dialog = SettingsWindow(self.hf_api)

        if dialog.exec():
            print("Updated Settings")

    def handle_update_page(self, index: int):
        self._view_model.select_data_by_index(index)

    def selecting_image(self, guid: UUID):
        print(f"Main -> selecting image guid: {guid}")

    def set_data(self, data: BudaOCRData):
        self.canvas.set_preview(data)
        self.text_view.update_text(data.ocr_text)
        self.current_guid = data.guid


class AppView(QWidget):
    def __init__(self, viewmodel: BudaViewModel, max_width: int, max_height: int):
        super().__init__()
        self.setObjectName("MainWindow")
        self.setContentsMargins(0, 0, 0, 0)
        self.controller = ImageController()
        self._viewmodel = viewmodel

        self.image_gallery = ImageGallery(self._viewmodel, self.controller)
        self.main_container = MainView(self._viewmodel, self.controller)

        self.setMinimumWidth(int(max_width * 0.5))
        self.setMaximumWidth(max_width)
        self.setMinimumHeight(int(max_height * 0.5))
        self.setMaximumHeight(max_height)

        # build layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.image_gallery)
        self.main_layout.addWidget(self.main_container)

        # self.main_layout.addWidget(self.footer)
        self.setLayout(self.main_layout)

        # connect layout signals
        self.main_container.sign_handle_import.connect(self.handle_file_import)
        self.main_container.sign_handle_page_select.connect(self.select_page)
        self.main_container.sign_on_file_save.connect(self.save)
        self.main_container.sign_run_ocr.connect(self.run_ocr)

        self.text_exporter = TextExporter()
        self.app_images = {}
        self.current_image = None

        # inference sessions
        self.line_config = read_line_model_config(LINE_MODEL_PATH)
        self.line_detection = LineDetection(self.line_config)

        self.layout_config = read_layout_model_config(LAYOUT_MODEL_PATH)
        self.layout_detection = LayoutDetection(self.layout_config)

        self.ocr_inference = OCRInference("Models/OCR/Woodblock/config.json")

        self.ocr_settings = OCRSettings(
            k_factor=1.7
        )
        self.threadpool = QThreadPool()

        self.setStyleSheet("""

            background-color: #1d1c1c;
            color: #000000;
            
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
                    line_data=None,
                    preview=None
                )
                new_data[guid] = palmtree_data

        self._viewmodel.add_data(new_data)

    def save(self):
        if not len(self.app_images) > 0:
            info_box = NotificationDialog(
                "Saving Annotations",
                "The current project contains no data."
            )
            info_box.exec()

        else:
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setViewMode(QFileDialog.ViewMode.List)

            save_dir = dialog.getExistingDirectory()
            print(f"SaveDir: {save_dir}")

            if save_dir is not None and save_dir != "":
                for photi_data in self.app_images.values():

                    if len(photi_data.ocr_text) > 0:
                        out_file = f"{save_dir}/{photi_data.image_name}.txt"

                        with open(out_file, "w", encoding="utf-8") as f:
                            for ocr_text in photi_data.ocr_text:
                                f.write(f"{ocr_text}\n")

    def select_page(self, index: int):
        self.image_gallery.select_page(index)

    def run_ocr(self, guid: UUID):
        data = self._viewmodel.get_data_by_guid(guid)
        image = cv2.imread(data.image_path)
        line_mask = self.line_detection.predict(image)
        line_data = get_line_data(image, line_mask)
        line_images = extract_line_images(line_data, self.ocr_settings.k_factor)
        ocr_text = self.ocr_inference.run_batch(line_images)

        self._viewmodel.update_ocr_data(guid, ocr_text)
        self._viewmodel.update_page_data(guid, line_data, line_mask)



