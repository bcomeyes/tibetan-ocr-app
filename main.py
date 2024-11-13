
"""
- build for windows: nuitka --standalone --windows-console-mode=disable --plugin-enable=pyside6 --windows-icon-from-ico=logo.ico main.py
- debug build for windows: nuitka --standalone --plugin-enable=pyside6 --windows-icon-from-ico=logo.ico main.py
- build for macss: nuitka --standalone --plugin-enable=pyside6 --macos-create-app-bundle --macos-app-icon=logo.icns main.py
"""
import sys
from PySide6.QtCore import QPoint
from BDRC.MVVM.view import AppView
from BDRC.MVVM.model import OCRDataModel, SettingsModel
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
from Config import read_settings
from BDRC.Utils import get_screen_center, get_platform
from PySide6.QtWidgets import QApplication
from BDRC.Styles import DARK


if __name__ == "__main__":
    platform = get_platform()
    app = QApplication()
    app.setStyleSheet(DARK)

    app_settings, ocr_settings = read_settings()

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

    sys.exit(app.exec())