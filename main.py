
"""
- build for windows:
     nuitka --standalone --windows-console-mode=disable --output-dir=WindowsBuild --plugin-enable=pyside6 --windows-icon-from-ico=logo.ico --company-name=BDRC --product-name="Tibetan OCR App" --file-version=1.0 --product-version=1.0  --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py
- build for macos:
    nuitka --standalone --output-dir=OSXBuild --plugin-enable=pyside6 --company-name=BDRC --product-name="Tibetan OCR App" --file-version=1.0 --product-version=1.0 --macos-app-name="BDRC Tibetan OCR App" --macos-signed-app-name="io.bdrc.ocrapp" --macos-create-app-bundle --macos-app-icon=logo.icns --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py
- build for Linux:
    nuitka --standalone --onefile --output-dir="LinuxBuild" --plugin-enable=pyside6 --company-name=BDRC --product-name="Tibetan OCR App" --file-version=1.0 --product-version=1.0 --include-data-dir=./Assets=Assets --include-data-dir=./Models=Models main.py

- Debug build using pyinstaller:
    MacOs (Default): pyinstaller main.py --distpath="DebugBuild" --add-data="Assets:Assets" --add-data="Models:Models" 
    MacOs (e.g. Intel): pyinstaller main.py --distpath="DebugBuild" --target-arch="x86_64" --add-data="Assets:Assets" --add-data="Models:Models" 
Note:
    If you edit the resources.qrc file, make sure to recompile it by using: pyside6-rcc resources.qrc -o resources.py
"""

import os
import sys
from platformdirs import user_data_dir
from PySide6.QtCore import QPoint
from BDRC.MVVM.view import AppView
from BDRC.MVVM.model import OCRDataModel, SettingsModel
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
from BDRC.Utils import get_screen_center, get_platform, create_dir
from PySide6.QtWidgets import QApplication
from BDRC.Styles import DARK

APP_NAME = "BDRC_OCR"
APP_AUTHOR = "BDRC"


if __name__ == "__main__":
    platform = get_platform()
    execution_dir= os.path.dirname(__file__)
    udi = user_data_dir(APP_NAME, APP_AUTHOR)
    create_dir(udi)
    
    app = QApplication()
    app.setStyleSheet(DARK)

    data_model = OCRDataModel()
    settings_model = SettingsModel(udi, execution_dir)

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
