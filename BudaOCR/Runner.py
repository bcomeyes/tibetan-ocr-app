import os
import cv2
from uuid import UUID
from typing import List
from BudaOCR.Data import OpStatus, OCResult, LineMode, BudaOCRData, Encoding, OCRSettings, OCRSample
from PySide6.QtCore import QObject, Signal, QRunnable

from BudaOCR.Inference import OCRPipeline
from BudaOCR.Utils import get_filename, generate_guid


class RunnerSignals(QObject):
    sample = Signal(OCRSample)
    error = Signal(str)
    finished = Signal()
    ocr_result = Signal(OCResult)
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
                    lines=None,
                    preview=None,
                    angle=0.0
                )

                imported_data[guid] = ocr_data

        self.signals.ocr_data.emit(imported_data)


class OCRunner(QRunnable):
    def __init__(self, data: BudaOCRData, ocr_pipeline: OCRPipeline, settings: OCRSettings):
        super(OCRunner, self).__init__()
        self.signals = RunnerSignals()
        self.data = data
        self.pipeline = ocr_pipeline
        self.settings = settings

    def run(self):
        img = cv2.imread(self.data.image_path)
        status, result = self.pipeline.run_ocr(img)

        if status == OpStatus.SUCCESS:
            print(f"Runner -> Done")
            rot_mask, lines, page_text, angle = result

            ocr_result = OCResult(
                guid=self.data.guid,
                mask=rot_mask,
                lines=lines,
                text=page_text,
                angle=angle
            )
            print(f"Runner->Emitting Signals...")
            self.signals.ocr_result.emit(ocr_result)
        else:
            print(f"Runner -> Failed")
            self.signals.finished()


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
            if not self.stop:
                img = cv2.imread(data.image_path)
                status, result = self.ocr_pipeline.run_ocr(img)

                if status == OpStatus.SUCCESS:
                    rot_mask, lines, page_text, angle = result

                    ocr_result = OCResult(
                        guid=data.guid,
                        mask=rot_mask,
                        lines=lines,
                        text=page_text,
                        angle=angle
                    )
                    results[data.guid] = ocr_result
                    sample = OCRSample(
                        cnt=idx,
                        guid=data.guid,
                        name=data.image_name,
                        result=ocr_result
                    )
                    self.signals.sample.emit(sample)

            else:
                print("Interrupted Process...")
                self.signals.finished.emit()

        self.signals.finished.emit()


