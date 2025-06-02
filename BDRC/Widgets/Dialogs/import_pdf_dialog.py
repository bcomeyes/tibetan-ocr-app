from PySide6.QtWidgets import QFileDialog

class PDFImportDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setNameFilter("PDF Files (*.pdf);;All Files (*)")
        self.setWindowTitle("Select PDF File")
