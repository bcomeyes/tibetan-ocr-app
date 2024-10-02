import os
import cv2
import json
import numpy as np
import onnxruntime as ort
import numpy.typing as npt
from scipy.special import softmax
from pyctcdecode import build_ctcdecoder
from BudaOCR.Utils import (
    pad_image,
    patch_image,
    sigmoid,
    normalize,
    unpatch_image,
    resize_to_width,
    resize_to_height,
    post_process_prediction,
    filter_contours,
    generate_line_preview,
    build_line_data,
    sort_lines_by_threshold,
    extract_line,
    binarize,
    prepare_ocr_line, calculate_steps, calculate_paddings, generate_patches, pad_to_width, pad_to_height,
)


from BudaOCR.Data import OpStatus, LineDetectionConfig, LayoutDetectionConfig

class Detection:
    def __init__(self, config: LineDetectionConfig | LayoutDetectionConfig):
        self.config = config
        self._config_file = config
        self._onnx_model_file = config.model_file
        self._patch_size = config.patch_size
        self._execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._inference = ort.InferenceSession(
            self._onnx_model_file, providers=self._execution_providers
        )

    def _prepare_image(
        self,
        image: npt.NDArray,
        patch_size: int = 512,
        resize: bool = True,
        fix_height: bool = False,
    ):
        if resize:
            if not fix_height:
                if image.shape[0] > 1.5 * patch_size:
                    target_height = 2 * patch_size
                else:
                    target_height = patch_size
            else:
                target_height = patch_size

            image, _ = resize_to_height(image, target_height=target_height)

        steps_x, steps_y = calculate_steps(image, patch_size)

        pad_x, pad_y = calculate_paddings(image, steps_x, steps_y, patch_size)
        padded_img = pad_image(image, pad_x, pad_y)
        print(f"Paded img: {padded_img.shape}")

        image_patches = generate_patches(padded_img, steps_x, steps_y, patch_size)
        image_patches = [binarize(x) for x in image_patches]

        image_patches = [normalize(x) for x in image_patches]
        image_patches = np.array(image_patches)

        return padded_img, image_patches, steps_y, steps_x, pad_x, pad_y

    def _unpatch_image(self, pred_batch: npt.NDArray, y_steps: int):
        # TODO: add some dimenions range and sequence checking so that things don't blow up when the input is not BxHxWxC
        dimensional_split = np.split(pred_batch, y_steps, axis=0)
        x_stacks = []

        for _, x_row in enumerate(dimensional_split):
            x_stack = np.hstack(x_row)
            x_stacks.append(x_stack)

        concat_out = np.vstack(x_stacks)

        return concat_out

    def _adjust_prediction(
        self, image: npt.NDArray, prediction: npt.NDArray, x_pad: int, y_pad: int
    ) -> npt.NDArray:
        x_lim = prediction.shape[1] - x_pad
        y_lim = prediction.shape[0] - y_pad

        prediction = prediction[:y_lim, :x_lim]
        prediction = cv2.resize(prediction, dsize=(image.shape[1], image.shape[0]))

        return prediction

    def _post_process(self, image: npt.NDArray) -> npt.NDArray:
        image = image.astype(np.uint8)
        image *= 255

        return image

    def _predict(self, image_batch: npt.NDArray):
        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()

        return prediction

    def predict(
        self, image: npt.NDArray, class_threshold: float = 0.8, fix_height: bool = True
    ) -> npt.NDArray:
        pass

class LineDetection(Detection):
    def __init__(self, config: LineDetectionConfig) -> None:
        super().__init__(config)

    def predict(
        self, image: npt.NDArray, class_threshold: float = 0.8, fix_height: bool = True
    ) -> npt.NDArray:

        _, image_patches, y_steps, x_steps, pad_x, pad_y = self._prepare_image(
            image, patch_size=self._patch_size, fix_height=fix_height
        )
        prediction = self._predict(image_patches)
        prediction = np.squeeze(prediction, axis=1)
        prediction = sigmoid(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        merged_image = self._unpatch_image(prediction, y_steps=y_steps)
        merged_image = self._adjust_prediction(image, merged_image, pad_x, pad_y)
        merged_image = self._post_process(merged_image)
        return merged_image


class LayoutDetection(Detection):
    def __init__(self, config: LayoutDetectionConfig) -> None:
        super().__init__(config)
        self._classes = config.classes
        print(f"Layout Classes: {self._classes}")

    def predict(
        self, image: npt.NDArray, class_threshold: float = 0.8, fix_height: bool = False
    ) -> npt.NDArray:
        _, image_patches, y_steps, x_steps, pad_x, pad_y = self._prepare_image(
            image, patch_size=self._patch_size, fix_height=fix_height
        )
        prediction = self._predict(image_patches)
        prediction = np.transpose(prediction, axes=[0, 2, 3, 1])
        prediction = softmax(prediction, axis=-1)
        prediction = np.where(prediction > class_threshold, 1.0, 0)
        merged_image = self._unpatch_image(prediction, y_steps)
        merged_image = self._adjust_prediction(image, merged_image, pad_x, pad_y)
        merged_image = self._post_process(merged_image)

        return merged_image



class OCRInference:
    def __init__(self, config_file, mode: str = "cuda") -> None:
        self.config = config_file
        self._onnx_model_file = None
        self._input_width = 2000
        self._input_height = 80
        self._characters = []
        self._ctcdecoder = None
        self._can_run = False
        self.ocr_session = None
        self.mode = mode
        self._init()

    def _init(self) -> None:
        _model_dir = os.path.dirname(self.config)
        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())
        self._onnx_model_file = f"{_model_dir}/{json_content['onnx-model']}"

        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._input_layer = json_content["input_layer"]
        self._output_layer = json_content["output_layer"]
        self._squeeze_channel_dim = (
            True if json_content["squeeze_channel_dim"] == "yes" else False
        )
        self._swap_hw = True if json_content["swap_hw"] == "yes" else False
        self._characters = self.get_charset(json_content["charset"])
        self._ctcdecoder = build_ctcdecoder(self._characters)

        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=execution_providers
        )

    def get_charset(self, charset: str) -> list[str]:
        charset = f"ß{charset}"
        charset = [x for x in charset]
        return charset

    def _pad_ocr_line(
            self,
            img: npt.NDArray,
            padding: str = "black",
    ) -> npt.NDArray:

        width_ratio = self._input_width / img.shape[1]
        height_ratio = self._input_height / img.shape[0]

        if width_ratio < height_ratio:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        elif width_ratio > height_ratio:
            out_img = pad_to_height(img, self._input_width, self._input_height, padding)
        else:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        return cv2.resize(
            out_img,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _prepare_ocr_line(self, image: npt.NDArray) -> npt.NDArray:
        line_image = self._pad_ocr_line(image)
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        line_image = line_image.reshape((1, self._input_height, self._input_width))
        line_image = (line_image / 127.5) - 1.0
        line_image = line_image.astype(np.float32)

        return line_image

    def _predict(self, image_batch: np.array):
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )

        logits = ocr_results[0].numpy()
        logits = np.squeeze(logits)

        return logits

    def _decode(self, logits: np.array) -> str:
        text = self._ctcdecoder.decode(logits)
        text = text.replace(" ", "")
        text = text.replace("§", " ")

        return text

    def run(self, line_image: np.array):

        if self._swap_hw:
            line_image = np.transpose(line_image, axes=[0, 2, 1])

        logits = self._predict(line_image)

        # This flag basically takes into account the different
        # ordering of the channels for CRNN and EASTER models. Maybe rename this variable or make this process
        # more explicit
        if not self._swap_hw:
            logits = np.transpose(logits, axes=[1, 0, 2])

        text = self._decode(logits)

        return text, logits

    def run_batch(self, line_images: list[np.array]):
        line_images = [self._prepare_ocr_line(x) for x in line_images]
        img_batch = np.array(line_images, np.float32)
        img_batch = np.squeeze(img_batch)

        if self._swap_hw:
            img_batch = np.transpose(img_batch, axes=[0, 2, 1])

        print(f"ImageBatch: {img_batch.shape}")
        logits = self._predict(img_batch)

        predicted_text = []

        for idx in range(logits.shape[0]):
            _logits = logits[idx, :, :]
            text = self._decode(_logits)
            predicted_text.append(text)

        return predicted_text


        """
        if self._swap_hw:
            line_image = np.transpose(img_batch, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            line_image = np.expand_dims(line_image, axis=1)

       
        text = self._decode(logits)

        img_batch = np.array(line_images, np.float32)

        if self._squeeze_channel_dim:
            img_batch = np.squeeze(img_batch, axis=1)

        if self._swap_hw:
            img_batch = np.transpose(img_batch, axes=[0, 2, 1])

        logits = self._predict(img_batch)

        predicted_text = []

        for idx in range(logits.shape[0]):
            _logits = logits[idx, :, :]
            text = self._decode(_logits)

            if len(text) > 0:
                text = text.replace(" ", "")
                text = text.replace("§", " ")
                predicted_text.append(text)

            if len(text) > 0:
                text = text.replace(" ", "")
                text = text.replace("§")
                predicted_text.append(text)

        return predicted_text
        """
