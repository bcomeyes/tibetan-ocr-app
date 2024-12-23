
"""
- build for windows: nuitka --standalone --windows-console-mode=disable --plugin-enable=pyside6 --windows-icon-from-ico=logo.ico --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py
- debug build for windows: nuitka --standalone --plugin-enable=pyside6 --windows-icon-from-ico=logo.ico --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py
- - build for macos: nuitka --standalone --plugin-enable=pyside6 --macos-create-app-bundle --macos-app-icon=logo.icns --include-data-dir=./Resources/Assets=Resources/Assets --include-data-dir=./Resources/Models=Resources/Models --output-dir="Build" main.py

Note:
    If you edit the resources.qrc file, make sure to recompile it by using: pyside6-rcc resources.qrc -o resources.py
"""

import os
import sys
from appdirs import user_data_dir
from PySide6.QtCore import QPoint
from BDRC.MVVM.view import AppView
from BDRC.MVVM.model import OCRDataModel, SettingsModel
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
from BDRC.Utils import get_screen_center, get_platform, create_dir
from PySide6.QtWidgets import QApplication
from BDRC.Styles import DARK

APP_NAME = "BDRC"
APP_AUTHOR = "OCR"


if __name__ == "__main__":
    platform = get_platform()
    execution_dir= os.path.dirname(__file__)
    user_data_dir = user_data_dir(APP_NAME, APP_AUTHOR)
    create_dir(user_data_dir)
    
    app = QApplication()
    app.setStyleSheet(DARK)

    data_model = OCRDataModel()
    settings_model = SettingsModel(user_data_dir, execution_dir)

    dataview_model = DataViewModel(data_model)
    settingsview_model = SettingsViewModel(settings_model)

    screen_data = get_screen_center(app)

    app_view = AppView(
        dataview_model,
        settingsview_model,
        platform
    )

    app_view.resize(screen_data.start_width, screen_data.start_height)
    app_view.move(QPoint(screen_data.start_x, screen_data.start_y))

    sys.exit(app.exec())
