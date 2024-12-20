import os
import json
import logging
from BDRC.Data import OpStatus, AppSettings, Encoding, ExportFormat, Language, Theme, OCRSettings, LineMode, \
    LineMerge, LineSorting, TPSMode, CharsetEncoder, OCRArchitecture
from huggingface_hub import snapshot_download


LINES_CONFIG = "Models/Lines/config.json"
LAYOUT_CONFIG = "Models/Layout/config.json"
DEFAULT_FONT = "Assets/Fonts/TibMachUni-1.901b.ttf"

"""
Mappings for each data type
"""

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

LANGUAGES = {
    "en": Language.English,
    "de": Language.German,
    "fr": Language.French,
    "bo": Language.Tibetan,
    "ch": Language.Chinese
}

ENCODINGS = {
    "unicode": Encoding.Unicode,
    "wylie": Encoding.Wylie
}

CHARSETENCODER = {
    "wylie": CharsetEncoder.Wylie,
    "stack": CharsetEncoder.Stack
}

OCRARCHITECTURE = {
    "Easter2": OCRArchitecture.Easter2,
    "CRNN": OCRArchitecture.CRNN
}
THEMES = {
    "dark": Theme.Dark,
    "light": Theme.Light
}

EXPORTERS = {
    "xml": ExportFormat.XML,
    "json": ExportFormat.JSON,
    "text": ExportFormat.Text
}

LINE_MODES = {
    "line": LineMode.Line,
    "layout": LineMode.Layout
}

LINE_MERGE = {
    "merge": LineMerge.Merge,
    "stack": LineMerge.Stack
}

LINE_SORTING = {
    "threshold": LineSorting.Threshold,
    "peaks": LineSorting.Peaks
}

TPS_MODE = {
    "local": TPSMode.LOCAL,
    "global": TPSMode.GLOBAL
}

DEFAULT_MODELS_PATH = "Models"
USER_MODEL_PATH = "/"

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


def read_settings(user_dir: str):
    app_settings_file = os.path.join(user_dir, "app_settings.json")
    ocr_settings_file = os.path.join(user_dir, "ocr_settings.json")

    if not os.path.isfile(app_settings_file):
        print("Creating Default app settings...")
        create_default_app_config(user_dir)

    if not os.path.isfile(ocr_settings_file):
        create_default_ocr_config(user_dir)

    file = open(app_settings_file, encoding="utf-8")
    app_json_settings = json.loads(file.read())

    _model_path = app_json_settings["model_path"]
    _lang_code = app_json_settings["language"]
    _encoding = app_json_settings["encoding"]
    _theme = app_json_settings["theme"]

    app_settings = AppSettings(
        model_path=_model_path,
        language=LANGUAGES[_lang_code],
        encoding=ENCODINGS[_encoding],
        theme=THEMES[_theme]
    )

    file = open(ocr_settings_file, encoding="utf-8")
    ocr_json_settings = json.loads(file.read())
    _line_mode = ocr_json_settings["line_mode"]
    _line_merge =  ocr_json_settings["line_merge"]
    _line_sorting = ocr_json_settings["line_sorting"]
    _k_factor = ocr_json_settings["k_factor"]
    _bbox_tolerance = ocr_json_settings["bbox_tolerance"]
    _dewarping = ocr_json_settings["dewarp"]
    _merge_lines = ocr_json_settings["merge_lines"]
    _tps = ocr_json_settings["tps"]
    _out_encoding = ocr_json_settings["output_encoding"]

    ocr_settings = OCRSettings(
        line_mode=LINE_MODES[_line_mode],
        line_merge=LINE_MERGE[_line_merge],
        line_sorting=LINE_SORTING[_line_sorting],
        dewarping=True if _dewarping == "yes" else False,
        merge_lines=True if _merge_lines == "yes" else False,
        k_factor=float(_k_factor),
        bbox_tolerance=float(_bbox_tolerance),
        tps_mode=TPS_MODE[_tps],
        output_encoding=ENCODINGS[_out_encoding],
    )

    return app_settings, ocr_settings


def save_app_settings(settings: AppSettings, user_dir: str):
    _model_path = settings.model_path
    _language = [x for x in LANGUAGES if LANGUAGES[x] == settings.language][0]
    _encoding = [x for x in ENCODINGS if ENCODINGS[x] == settings.encoding][0]
    _theme = [x for x in THEMES if THEMES[x] == settings.theme][0]

    _settings = {
                "model_path": _model_path,
                "language": _language,
                "encoding": _encoding,
                "theme": _theme
            }

    app_settings_file = os.path.join(user_dir, "app_settings.json")
    with open(app_settings_file, "w", encoding="utf-8") as f:
        json.dump(_settings, f, ensure_ascii=False, indent=1)


def save_ocr_settings(settings: OCRSettings, user_dir: str):
    _line_mode = [x for x in LINE_MODES if LINE_MODES[x] == settings.line_mode][0]
    _line_merge = [x for x in LINE_MERGE if LINE_MERGE[x] == settings.line_merge][0]
    _line_sorting = [x for x in LINE_SORTING if LINE_SORTING[x] == settings.line_sorting][0]
    _dewarp = "yes" if settings.dewarping else "no"
    _merge_lines = "yes" if settings.merge_lines else "no"
    _tps = [x for x in TPS_MODE if TPS_MODE[x] == settings.tps_mode][0]

    _settings = {
        "line_mode": _line_mode,
        "line_merge": _line_merge,
        "line_sorting": _line_sorting,
        "dewarp": _dewarp,
        "merge_lines": _merge_lines,
        "k_factor": settings.k_factor,
        "bbox_tolerance": settings.bbox_tolerance,
        "tps": _tps,
        "output_encoding": "unicode",
    }

    ocr_settings_file = os.path.join(user_dir, "ocr_settings.json")
    with open(ocr_settings_file, "w", encoding="utf-8") as f:
        json.dump(_settings, f, ensure_ascii=False, indent=1)


def create_default_app_config(user_dir: str):
    _settings = {
            "model_path": "Models",
            "language": "en",
            "encoding": "unicode",
            "theme": "dark"
        }
    app_settings_file = os.path.join(user_dir, "app_settings.json")
    with open(app_settings_file, "w", encoding="utf-8") as f:
        json.dump(_settings, f, ensure_ascii=False, indent=1)


def create_default_ocr_config(user_dir: str):
    _settings = {
        "line_mode": "line",
        "line_merge": "merge",
        "line_sorting": "threshold",
        "dewarp": "yes",
        "merge_lines": "yes",
        "k_factor": 2.5,
        "bbox_tolerance": 2.5,
        "tps": "global",
        "output_encoding": "unicode"
    }

    ocr_settings_file = os.path.join(user_dir, "ocr_settings.json")
    with open(ocr_settings_file, "w", encoding="utf-8") as f:
        json.dump(_settings, f, ensure_ascii=False, indent=1)
