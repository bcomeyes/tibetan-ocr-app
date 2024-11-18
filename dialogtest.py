import sys
from glob import glob
from typing import List
from PySide6.QtCore import QThreadPool, QRunnable, QObject, Signal, QSize, Qt
from PySide6.QtWidgets import QApplication, QProgressBar, QPushButton, QVBoxLayout, QDialog, QListWidgetItem, QWidget, \
    QLabel

from BDRC.Widgets.Layout import ImageListWidget
from BDRC.Utils import generate_guid

class ImportRunnerSignals(QObject):
    s_sample_count = Signal(int)
    s_finished = Signal()

class ImportFilesProgress(QDialog):
    def __init__(self, max_length: int):
        super().__init__()
        self.setObjectName("OCRDialog")
        self.setWindowTitle("Importing Files...")
        self.progress_canceled = False

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(max_length)

        self.setFixedHeight(180)
        self.setFixedWidth(420)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.setFixedHeight(48)
        self.cancel_btn.setObjectName("DialogButton")

        self.cancel_btn.clicked.connect(self.handle_cancel)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.cancel_btn)
        self.setLayout(self.layout)

        self.cancel_btn.setStyleSheet("""

                QPushButton {
                    background-color: #ff0000;
                }

                QPushButton::hover {
                    color: #ffad00;
                }

            """)

        self.setStyleSheet("""
            background-color: #08081f;

            QProgressBar {
                background-color: #24272c;
                border-radius: 5px;
                border-width: 2px;
            }

            QProgressBar::chunk
            {
                background-color: #003d66;
                border-radius: 5px;
                margin: 3px 3px 3px 3px;
            }
        """)

        self.show()

    def handle_cancel(self):
        self.progress_canceled = True

    def set_value(self, value: int):
        self.progress_bar.setValue(value)


if __name__ == "__main__":
    DIR = "C:/Users/Eric/Desktop/ImportTest"
    images = glob(f"{DIR}/*.jpg")

    app = QApplication()
    dialog = ImportFilesProgress(120)
    dialog.exec()

    sys.exit(app.exec())