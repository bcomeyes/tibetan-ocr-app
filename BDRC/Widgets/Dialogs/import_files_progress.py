from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QPushButton, QProgressBar

class ImportFilesProgress(QProgressDialog):
    def __init__(self, title: str, max_length: int = 100):
        super(ImportFilesProgress, self).__init__()
        self.setWindowTitle(title)
        self.setFixedWidth(420)
        self.setFixedHeight(140)
        self.setContentsMargins(10, 10, 10, 10)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
                color: #ffffff;
                background-color: #A40021;
                border-radius: 4px;
                height: 20;
                width: 80px;
                margin-top: 10px;
        """)

        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.setFixedHeight(32)
        self.setCancelButton(self.cancel_btn)

        self.cancel_btn.clicked.connect(self.cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(max_length)
        self.progress_bar.setObjectName("DialogProgressBar")

        self.setBar(self.progress_bar)
