from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QPushButton

class NotificationDialog(QMessageBox):
    def __init__(self, title: str, message: str):
        super().__init__()
        self.setObjectName("NotificationWindow")
        self.setWindowTitle(title)
        self.setFixedWidth(320)
        self.setFixedHeight(120)
        self.setIcon(QMessageBox.Icon.Information)

        self.setText(message)

        self.ok_btn = QPushButton("Ok")
        self.ok_btn.setStyleSheet("""
           
            color: #ffffff;
            background-color: #A40021;
            border-radius: 4px;
            height: 20;
            width: 80px;

            QPushButton::hover { 
               color: #ff0000;
           }
        """
        )

        self.addButton(self.ok_btn, QMessageBox.ButtonRole.YesRole)

        self.setStyleSheet("""
            color: #ffffff;
            background-color: #1d1c1c;
        """)
