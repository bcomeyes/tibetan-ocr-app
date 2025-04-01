from PySide6.QtWidgets import QMessageBox, QPushButton
from PySide6.QtCore import Qt

class ConfirmationDialog(QMessageBox):
    def __init__(self, title: str, message: str, show_cancel: bool = True):
        super().__init__()
        self.setObjectName("ConfirmWindow")
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self.setMinimumHeight(440)
        self.setIcon(QMessageBox.Icon.Information)
        self.setText(message)

        self.ok_btn = QPushButton("Ok")
        self.cancel_btn = QPushButton("Cancel")

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.ok_btn.setStyleSheet(
            """
                color: #ffffff;
                font: bold 12px;
                width: 240px;
                height: 32px;
                background-color: #A40021;
                border: 2px solid #A40021;
                border-radius: 4px;

                QPushButton::hover { 
                    color: #ff0000;
                }

            """
        )

        self.cancel_btn.setStyleSheet(
            """
                color: #ffffff;
                font: bold 12px;
                width: 240px;
                height: 32px;
                background-color: #A40021;
                border: 2px solid #A40021;
                border-radius: 4px;

                QPushButton::hover {
                    color: #ff0000;
                }
            """
        )

        if show_cancel:
            self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)
            self.addButton(self.cancel_btn, QMessageBox.ButtonRole.NoRole)
        else:
            self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)

        self.setStyleSheet("""
                    color: #ffffff;
                    background-color: #1d1c1c;
            """)
