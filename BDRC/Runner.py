import cv2
from uuid import UUID
from typing import List
from PySide6.QtCore import QObject, Signal, QRunnable

from BDRC.Inference import OCRPipeline
from BDRC.Data import OpStatus, OCResult, LineMode, OCRData, Encoding, OCRSettings, OCRSample



class RunnerSignals(QObject):
    sample = Signal(OCRSample)
    sample_count = Signal(int)
    error = Signal(str)
    finished = Signal()
    ocr_result = Signal(OCResult)
    ocr_data = Signal(dict[UUID, OCRData])

class OCRunner(QRunnable):
    def __init__(self, data: OCRData, ocr_pipeline: OCRPipeline, settings: OCRSettings):
        super(OCRunner, self).__init__()
        self.signals = RunnerSignals()
        self.data = data
        self.pipeline = ocr_pipeline
        self.settings = settings
        self.k_factor =  settings.k_factor
        self.bbox_tolerance = settings.bbox_tolerance

    def run(self):
        img = cv2.imread(self.data.image_path)
        status, result = self.pipeline.run_ocr(img, k_factor=self.k_factor, bbox_tolerance=self.bbox_tolerance)

        if status == OpStatus.SUCCESS:
            rot_mask, lines, page_text, angle = result
            print(f"OCRunner -> Result: lines: {len(lines)}, Textlines: {len(page_text)}")
            ocr_result = OCResult(
                guid=self.data.guid,
                mask=rot_mask,
                lines=lines,
                text=page_text,
                angle=angle
            )
            self.signals.ocr_result.emit(ocr_result)
        else:
            self.signals.finished.emit()


class OCRBatchRunner(QRunnable):
    def __init__(
            self,
            data: List[OCRData],
            ocr_pipeline: OCRPipeline,
            mode: LineMode = LineMode.Layout,
            dewarp: bool = True,
            merge_lines: bool = True,
            k_factor: float = 1.7,
            bbox_tolerance: float = 3.0,
            ):

        super(OCRBatchRunner, self).__init__()
        self.signals = RunnerSignals()
        self.data = data
        self.ocr_pipeline = ocr_pipeline
        self.data = data
        self.mode = mode
        self.do_dewarp = dewarp
        self.merge_lines = merge_lines
        self.k_factor = k_factor
        self.bbox_tolerance = bbox_tolerance
        self.stop = False

    def kill(self):
        print(f"OCRunner -> kill")
        self.stop = True

    def run(self):
        results = {}

        for idx, data in enumerate(self.data):
            if not self.stop:
                img = cv2.imread(data.image_path)
                status, result = self.ocr_pipeline.run_ocr(
                    image=img,
                    k_factor=self.k_factor,
                    bbox_tolerance=self.bbox_tolerance,
                    merge_lines=self.merge_lines,
                    use_tps=self.do_dewarp
                )

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
                self.signals.finished.emit()

        self.signals.finished.emit()


