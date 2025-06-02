from PySide6.QtWidgets import QFileDialog

class ImportImagesDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)")
        self.setWindowTitle("Select Image Files")
