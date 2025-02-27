import os
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap, QIcon, QColor, QPalette
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSizePolicy, QApplication
)

class ClickableOptionPanel(QFrame):
    """A clickable panel that can be selected."""
    
    clicked = Signal(int)
    
    def __init__(self, option_id, title, description, parent=None):
        super().__init__(parent)
        self.option_id = option_id
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setCursor(Qt.PointingHandCursor)
        
        # Set up the layout
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        
        # Description
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #ffffff;")
        
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        
        # Set minimum size
        self.setMinimumHeight(120)
        
        # Style
        self.setStyleSheet("""
            ClickableOptionPanel {
                border: 1px solid #555555;
                border-radius: 5px;
                background-color: #2d2d2d;
                padding: 15px;
            }
            ClickableOptionPanel:hover {
                background-color: #3d3d3d;
                border: 1px solid #777777;
            }
        """)
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.option_id)
        super().mousePressEvent(event)

class PDFImportDialog(QDialog):
    """Dialog that allows users to choose between different PDF import methods."""
    
    # Define import methods as constants
    IMPORT_EMBEDDED_IMAGES = 1
    IMPORT_CONVERT_PAGES = 2
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PDF Import Options")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.selected_method = None
        self._setup_dark_theme()
        self._setup_ui()
        
    def _setup_dark_theme(self):
        """Set up dark theme for the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #0d6efd;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #0a58ca;
            }
            QPushButton#cancelButton {
                background-color: #6c757d;
            }
            QPushButton#cancelButton:hover {
                background-color: #5c636a;
            }
        """)
        
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Title and description
        title_label = QLabel("Choose PDF Import Method")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        
        description_label = QLabel(
            "Select how you want to import PDF files into the application. "
            "Each method has different advantages depending on your PDF content."
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #cccccc;")
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        main_layout.addSpacing(20)
        
        # Option 1: Extract embedded images
        embedded_description = (
            "<b>Best for:</b> PDFs with pre-scanned images.<br>"
            "<b>Advantages:</b> Faster, preserves original image quality.<br>"
            "<b>Limitations:</b> Only works if the PDF contains embedded images."
        )
        
        option1_panel = ClickableOptionPanel(
            self.IMPORT_EMBEDDED_IMAGES,
            "Extract Embedded Images",
            embedded_description,
            self
        )
        option1_panel.clicked.connect(self._on_option_clicked)
        
        # Option 2: Convert pages to images
        convert_description = (
            "<b>Best for:</b> Any PDF, including those with text or vector graphics.<br>"
            "<b>Advantages:</b> Works with all PDFs, consistent results.<br>"
            "<b>Limitations:</b> May be slower for large documents."
        )
        
        option2_panel = ClickableOptionPanel(
            self.IMPORT_CONVERT_PAGES,
            "Convert Pages to Images",
            convert_description,
            self
        )
        option2_panel.clicked.connect(self._on_option_clicked)
        
        # Add options to layout
        main_layout.addWidget(option1_panel)
        main_layout.addSpacing(15)
        main_layout.addWidget(option2_panel)
        main_layout.addStretch()
        
        # Cancel button at the bottom
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("cancelButton")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        
        main_layout.addLayout(button_layout)
    
    def _on_option_clicked(self, option_id):
        """Handle option panel click."""
        self.selected_method = option_id
        self.accept()
    
    def get_selected_method(self):
        """Return the selected import method."""
        return self.selected_method
