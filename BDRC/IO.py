import logging
from BDRC.Data import OpStatus, OCRData


class TextExporter:
    def __init__(self):
        pass

    def export(self, save_dir: str, data: list[OCRData]) -> OpStatus:

        for _data in data:
            if len(_data.ocr_text) > 0:
                out_file = f"{save_dir}/{_data.image_name}.txt"

                try:
                    with open(out_file, "w", encoding="utf-8") as f:
                        for ocr_text in _data.ocr_text:
                            f.write(f"{ocr_text}\n")

                except IOError as e:
                    logging.info(f"Saving File failed as IOError: {e}")
                    return OpStatus.FAILED

        return OpStatus.SUCCESS


