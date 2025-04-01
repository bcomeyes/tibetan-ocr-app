from PySide6.QtWidgets import QFileDialog

class PDFImportDialog(QFileDialog):
    def __init__(self, parent=None):
        super(PDFImportDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setNameFilter("PDF file (*.pdf)")
        self.setViewMode(QFileDialog.ViewMode.List)
