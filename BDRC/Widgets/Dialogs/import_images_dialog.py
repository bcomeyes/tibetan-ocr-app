from PySide6.QtWidgets import QFileDialog

class ImportImagesDialog(QFileDialog):
    def __init__(self, parent=None):
        super(ImportImagesDialog, self).__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.setNameFilter("Images (*.png *.jpg *.tif *.tiff)")
        self.setViewMode(QFileDialog.ViewMode.List)
