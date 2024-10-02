import os
import logging
from BudaOCR.Data import OpStatus
from huggingface_hub import snapshot_download


COLOR_DICT = {
        "background": "0, 0, 0",
        "image": "45, 255, 0",
        "text": "255, 243, 0",
        "margin": "0, 0, 255",
        "caption": "255, 100, 243",
        "table": "0, 255, 0",
        "pagenr": "0, 100, 15",
        "header": "255, 0, 0",
        "footer": "255, 255, 100",
        "line": "0, 100, 255"
    }


DEFAULT_PHOTI_MODEL = "BDRC/Photi"
DEFAULT_PHOTI_LOCAL_PATH = "Models/Photi/Default"


OCR_MODEL_STORE = {
    "Glomanthang": "Models/OCR/Glomanthang",
    "Woodblock": "Models/OCR/Woodblock",
    "Betsug": "Models/OCR/Betsug"
}

DEFAULT_OCR_LOCAL_PATH = "Models/OCR/Default"
DEFAULT_OCR_MODEL = "BDRC/Woodblock"


def init_models(model_dir: str = DEFAULT_PHOTI_LOCAL_PATH) -> OpStatus:
    _config_path = os.path.join(DEFAULT_PHOTI_LOCAL_PATH, "config.json")
    _model_path = os.path.join(DEFAULT_PHOTI_LOCAL_PATH, "photi.onnx")

    if not os.path.isdir(DEFAULT_PHOTI_LOCAL_PATH):
        try:
            snapshot_download(
                repo_id=DEFAULT_PHOTI_MODEL,
                repo_type="model",
                local_dir=model_dir,
            )
        except BaseException as e:
            logging.error(f"Failed to download default Photi model: {e}")
            return OpStatus.FAILED

        assert os.path.isfile(_config_path) and os.path.isfile(_model_path)

        return OpStatus.SUCCESS
    else:
        assert os.path.isfile(_config_path) and os.path.isfile(_model_path)
        return OpStatus.SUCCESS


def get_default_model() -> str:
    _config_path = os.path.join(DEFAULT_PHOTI_MODEL, "config.json")
    return _config_path


def init_default_ocr_model(model_dir: str = DEFAULT_OCR_LOCAL_PATH):
    _config_path = os.path.join(DEFAULT_OCR_LOCAL_PATH, "config.json")

    if not os.path.isdir(DEFAULT_OCR_LOCAL_PATH):
        try:
            snapshot_download(
                repo_id=DEFAULT_OCR_MODEL,
                repo_type="model",
                local_dir=model_dir,
            )
        except BaseException as e:
            logging.error(f"Failed to download default OCR model: {e}")

        assert os.path.isfile(_config_path)
    else:
        assert os.path.isfile(_config_path)