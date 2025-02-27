from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtGui import QFont

class TextInputDialog(QDialog):
    def __init__(self, title: str, edit_text: str, qfont: QFont, parent: QWidget | None):
        super(TextInputDialog, self).__init__()
        self.setObjectName("TextInputDialog")
        self.parent = parent
        self.title = title
        self.edit_text = edit_text
        self.new_text = ""
        self.qfont = qfont
        self.setMinimumWidth(480)
        self.setMinimumHeight(180)
        self.setWindowTitle(title)
        self.spacer = QLabel()
        self.spacer.setFixedHeight(36)

        self.line_edit = QLineEdit(self)
        self.line_edit.setObjectName("DialogLineEdit")
        self.line_edit.setFont(self.qfont)
        self.line_edit.setText(self.edit_text)
        self.line_edit.editingFinished.connect(self.update_text)

        self.accept_btn = QPushButton("Accept")
        self.reject_btn = QPushButton("Reject")
        self.accept_btn.setObjectName("SmallDialogButton")
        self.reject_btn.setObjectName("SmallDialogButton")

        self.accept_btn.clicked.connect(self.accept_change)
        self.reject_btn.clicked.connect(self.reject_change)

        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.accept_btn)
        self.h_layout.addWidget(self.reject_btn)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.line_edit)
        self.v_layout.addWidget(self.spacer)
        self.v_layout.addLayout(self.h_layout)

        self.setLayout(self.v_layout)

    def update_text(self):
        self.new_text = self.line_edit.text()

    def accept_change(self):
        self.accept()

    def reject_change(self):
        self.reject()
