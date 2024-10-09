import os
import cv2
from uuid import UUID
from typing import Dict, List
from BudaOCR.Data import LineData, LineDataResult, OCResult, LineMode, BudaOCRData
from PySide6.QtCore import QObject, Signal, QRunnable

from BudaOCR.Inference import LineDetection, LayoutDetection, OCRInference
from BudaOCR.Utils import binarize, get_line_data, extract_line, get_filename, generate_guid


class RunnerSignals(QObject):
    sample = Signal(int)
    error = Signal(str)
    finished = Signal()
    line_result = Signal(LineDataResult)
    ocr_result = Signal(OCResult)
    ocr_data = Signal(Dict[UUID, BudaOCRData])


class FileImportRunner(QRunnable):
    def __init__(self, files: List[str]):
        self.file_list = files
        self.signals = RunnerSignals()
        super(FileImportRunner, self).__init__()

    def run(self):
        imported_data = {}
        for idx, file_path in enumerate(self.file_list):
            if os.path.isfile(file_path):
                file_name = get_filename(file_path)
                guid = generate_guid(idx)
                ocr_data = BudaOCRData(
                    guid=guid,
                    image_path=file_path,
                    image_name=file_name,
                    ocr_text=[],
                    line_data=None,
                    preview=None
                )
                imported_data[guid] = ocr_data

        self.signals.ocr_data.emit(imported_data)


class OCRunner(QRunnable):
    def __init__(
            self,
            line_detection: LineDetection,
            layout_detection: LayoutDetection,
            ocr_inference: OCRInference,
            mode: LineMode,
            data: list[BudaOCRData],
            k_factor: float = 1.2):

        super(OCRunner, self).__init__()
        self.signals = RunnerSignals()
        self.line_detection = LineDetection
        self.layout_detection = LayoutDetection
        self.ocr_inference = OCRInference
        self.data = data
        self.mode = mode
        self.k_factor = k_factor

    def run(self):
        for idx, data in enumerate(self.data):
            image = cv2.imread(data.image_path)
            image = binarize(image)

            if self.mode == LineMode.Line:
                line_mask = self.line_detection.predict(image)
                line_data = get_line_data(image, line_mask)
            else:
                layout_mask = self.layout_detection.predict(image)
                line_data = get_line_data(
                    image, layout_mask[:, :, 2]
                )  # for the dim, see classes in the layout config file

            if len(line_data.lines) > 0:
                line_images = [extract_line(x, line_data.image, self.k_factor) for x in line_data.lines]

                page_text = []
                filtered_lines = []

                for line_img, line_info in zip(line_images, line_data.lines):
                    pred = self.ocr_inference.run(line_img)
                    pred = pred.strip()

                    if pred != "":
                        page_text.append(pred)
                        filtered_lines.append(line_info)

                filtered_line_data = LineData(
                    line_data.image, line_data.prediction, line_data.angle, filtered_lines
                )

                ocr_result = OCRResult(
                    data.guid,
                    page_text
                )

                line_result = LineDataResult(
                    data.guid,
                    filtered_line_data
                )
                self.signals.line_result.emit(line_result)
                self.signals.ocr_result.emit(ocr_result)

        self.signals.finished.emit()


