from uuid import UUID
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QPushButton
)


class ModelEntry(QWidget):
    def __init__(self, url: str, title: str, parent=None):
        super().__init__(parent)
        self.url = url
        self.title = title

        self.label = QLabel(self.title)
        self.download_btn = QPushButton('Download')
        self.delete_btn = QPushButton('Delete')

        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.label)
        self.h_layout.addWidget(self.download_btn)
        self.h_layout.addWidget(self.delete_btn)

        self.setLayout(self.h_layout)



class ModelEntryWidget(QWidget):
    def __init__(self, guid: UUID, title: str, encoder: str, architecture: str, version: str):
        super().__init__()
        self.guid = guid
        self.title = str(title)
        self.encoder = str(encoder)
        self.architecture = str(architecture)
        self.version = version

        self.title_label = QLabel(self.title)
        self.encoder_label = QLabel(self.encoder)
        self.architecture_label = QLabel(self.architecture)
        self.version_number =  QLabel(self.version)

        # build layout
        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.title_label)
        self.h_layout.addWidget(self.encoder_label)
        self.h_layout.addWidget(self.architecture_label)
        self.h_layout.addWidget(self.version_number)
        self.setLayout(self.h_layout)

        self.setStyleSheet("""
            color: #ffffff;
            width: 80%;
        """)

    def set_dark_background(self):
        self.setStyleSheet("""
            color: #ffffff;
            background-color: #242424;
        """)

    def set_light_background(self):
        self.setStyleSheet("""
            color: #ffffff;
            background-color: #3a3a3a;         
        """)
