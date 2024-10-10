import os
import cv2
from uuid import UUID
from typing import Dict, List
from BudaOCR.Data import OpStatus, LineDataResult, OCResult, LineMode, BudaOCRData, Encoding
from PySide6.QtCore import QObject, Signal, QRunnable

from BudaOCR.Inference import OCRPipeline
from BudaOCR.Utils import get_filename, generate_guid


class RunnerSignals(QObject):
    sample = Signal(int)
    error = Signal(str)
    finished = Signal()
    line_result = Signal(LineDataResult)
    ocr_result = Signal(OCResult)
    batch_ocr_result = Signal(dict[UUID, OCResult])
    ocr_data = Signal(dict[UUID, BudaOCRData])


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


class OCRBatchRunner(QRunnable):
    def __init__(
            self,
            data: List[BudaOCRData],
            ocr_pipeline: OCRPipeline,
            output_encoding: Encoding,
            mode: LineMode = LineMode.Layout,
            k_factor: float = 1.7):

        super(OCRBatchRunner, self).__init__()
        self.signals = RunnerSignals()
        self.data = data
        self.ocr_pipeline = ocr_pipeline
        self.data = data
        self.mode = mode
        self.k_factor = k_factor
        self.stop = False

    def kill(self):
        print(f"OCRunner -> kill")
        self.stop = True

    def run(self):
        results = {}

        for idx, data in enumerate(self.data):
            self.signals.sample.emit(idx)
            if not self.stop:
                img = cv2.imread(data.image_path)
                status, ocr_result = self.ocr_pipeline.run_ocr(img)

                if status == OpStatus.SUCCESS:
                    results[data.guid] = ocr_result
            else:
                print("Interrupted Process...")
                self.signals.finished.emit()

        self.signals.batch_ocr_result.emit(results)
        self.signals.finished.emit()


