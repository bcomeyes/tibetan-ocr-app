
"""
- build for windows: nuitka --standalone --windows-console-mode=disable --plugin-enable=pyside6 main.py
- build for macss: nuitka --standalone --plugin-enable=pyside6 --macos-create-app-bundle --macos-app-icon=logo.icns main.py
"""

import sys
from PySide6.QtCore import QPoint
from BudaOCR.MVVM.view import AppView
from BudaOCR.MVVM.model import BudaOCRDataModel
from BudaOCR.MVVM.viewmodel import BudaViewModel
from BudaOCR.Utils import get_screen_center
from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication()
    model = BudaOCRDataModel()
    view_model = BudaViewModel(model)

    screen_data = get_screen_center(app)
    app_view = AppView(view_model, screen_data.max_width, screen_data.max_height)
    app_view.resize(screen_data.start_width, screen_data.start_height)
    app_view.move(QPoint(0, 0))
    app_view.move(QPoint(screen_data.start_x, screen_data.start_y))
    sys.exit(app.exec())