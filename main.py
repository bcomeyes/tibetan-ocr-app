
"""
- build for windows: nuitka --standalone --windows-console-mode=disable --plugin-enable=pyside6 --windows-icon-from-ico=logo.ico --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py
- debug build for windows: nuitka --standalone --plugin-enable=pyside6 --windows-icon-from-ico=logo.ico --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py
- build for macos: nuitka --standalone --plugin-enable=pyside6 --macos-create-app-bundle --macos-app-icon=logo.icns --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py
"""
import os
import sys
from glob import glob
from PySide6.QtCore import QPoint
from BDRC.MVVM.view import AppView
from BDRC.MVVM.model import OCRDataModel, SettingsModel
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
from Config import TMP_DIR, read_settings
from BDRC.Utils import get_screen_center, get_platform, create_dir
from PySide6.QtWidgets import QApplication
from BDRC.Styles import DARK


if __name__ == "__main__":
    platform = get_platform()
    app = QApplication()
    app.setStyleSheet(DARK)

    app_settings, ocr_settings = read_settings()
    print(f"starting with model_path: {app_settings.model_path}")

    data_model = OCRDataModel()
    settings_model = SettingsModel(app_settings, ocr_settings)

    dataview_model = DataViewModel(data_model)
    settingsview_model = SettingsViewModel(settings_model)

    screen_data = get_screen_center(app)

    app_view = AppView(
        dataview_model,
        settingsview_model,
        platform,
        screen_data.max_width,
        screen_data.max_height
    )

    app_view.resize(screen_data.start_width, screen_data.start_height)
    app_view.move(QPoint(screen_data.start_x, screen_data.start_y))

    # just delete tmp files on startup
    create_dir(TMP_DIR)
    tmp_files = glob(f"{TMP_DIR}/*")

    if len(tmp_files) > 0:
        for file in tmp_files:
            os.remove(file)

    sys.exit(app.exec())