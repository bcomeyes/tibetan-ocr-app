from PySide6.QtWidgets import (
    QScrollBar,
    QWidget,
    QLabel,
    QSpacerItem,
    QHBoxLayout,
    QListView, QPushButton
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
