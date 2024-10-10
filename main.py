
"""
- build for windows: nuitka --standalone --windows-console-mode=disable --plugin-enable=pyside6 main.py
- build for macss: nuitka --standalone --plugin-enable=pyside6 --macos-create-app-bundle --macos-app-icon=logo.icns main.py
"""
import sys
import platform
from PySide6.QtCore import QPoint

from BudaOCR.Data import Language
from BudaOCR.MVVM.view import AppView
from BudaOCR.MVVM.model import BudaOCRDataModel, BudaSettingsModel
from BudaOCR.MVVM.viewmodel import BudaDataViewModel, BudaSettingsViewModel
from BudaOCR.Config import read_settings
from BudaOCR.Utils import get_screen_center
from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    platform = platform.platform()
    app = QApplication()
    app_settings, ocr_settings = read_settings()

    data_model = BudaOCRDataModel()
    settings_model = BudaSettingsModel(app_settings, ocr_settings)

    dataview_model = BudaDataViewModel(data_model)
    settingsview_model = BudaSettingsViewModel(settings_model)

    screen_data = get_screen_center(app)

    app_view = AppView(
        dataview_model,
        settingsview_model,
        screen_data.max_width,
        screen_data.max_height
    )

    app_view.resize(screen_data.start_width, screen_data.start_height)
    app_view.move(QPoint(0, 0))
    app_view.move(QPoint(screen_data.start_x, screen_data.start_y))

    sys.exit(app.exec())