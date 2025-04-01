from uuid import UUID
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QListView
)

class ModelListWidget(QWidget):
    def __init__(self, guid: UUID, title: str, encoder: str, architecture: str):
        super().__init__()
        self.setObjectName("QModelList")
        self.guid = guid
        self.title = str(title)
        self.encoder = str(encoder)
        self.architecture = str(architecture)

        self.title_label = QLabel(self.title)
        self.encoder_label = QLabel(self.encoder)
        self.architecture_label = QLabel(self.architecture)
        self.download_btn = QPushButton("Download")
        self.delete_btn = QPushButton("Delete")

        # build layout
        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.title_label)
        self.h_layout.addWidget(self.encoder_label)
        self.h_layout.addWidget(self.architecture_label)
        self.h_layout.addWidget(self.download_btn)
        self.h_layout.addWidget(self.delete_btn)
        self.setLayout(self.h_layout)


class ModelList(QListWidget):
    """This Widget is currently not used and is just here in case the table view in the SettingsDialog shall
    be replaced by a QListWidget that supports custom ListQWigets with delete buttons etc."""
    
    sign_on_selected_item = Signal(UUID)
    
    def __init__(self, parent=None):
        super(ModelList, self).__init__(parent)
        self.parent = parent
        self.setObjectName("ModelList")
        self.setFlow(QListView.Flow.TopToBottom)
        self.setMouseTracking(True)
        self.itemClicked.connect(self.on_item_clicked)
        #self.itemEntered.connect(self.on_item_entered)
    
    def on_item_clicked(self, item: QListWidgetItem):
        widget = self.itemWidget(item)
        if widget:
            self.sign_on_selected_item.emit(widget.guid)
