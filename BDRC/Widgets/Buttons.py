from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QEnterEvent, QPixmap, QColor
from PySide6.QtWidgets import QPushButton


class MenuButton(QPushButton):
    def __init__(
            self,
            hint: str,
            icon_path: str,
            width: int = 40,
            height: int = 40,
            object_name: str = "MenuButton",
            parent=None):

        super().__init__()
        self.parent = parent
        self.setObjectName(object_name)
        self.width = width
        self.height = height
        self._icon_path = icon_path
        self.hint = hint
        self._pixmap = QPixmap(self._icon_path)

        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)
        self.setToolTip(self.hint)
        self.default_color = "#ffffff"
        self.highlight_color = "#F2CD9B"
        self.is_active = False

        self.set_default_icon()

    def set_hover_icon(self) -> None:
        mask = self._pixmap.createMaskFromColor(QColor('transparent'), Qt.MaskMode.MaskInColor)
        self._pixmap.fill((QColor(self.highlight_color)))
        self._pixmap.setMask(mask)
        self.setIcon(QIcon(self._pixmap))

    def set_default_icon(self):
        mask = self._pixmap.createMaskFromColor(QColor('transparent'), Qt.MaskMode.MaskInColor)
        self._pixmap.fill((QColor('white')))
        self._pixmap.setMask(mask)

        self.setIcon(QIcon(self._pixmap))

    def activate(self) -> None:
        self.is_active = True
        self.set_default_icon()

    def deactivate(self) -> None:
        self.is_active = False
        self.set_default_icon()

    def enterEvent(self, event):
        if isinstance(event, QEnterEvent):
            self.set_hover_icon()
            return super().enterEvent(event)

    def leaveEvent(self, event):
        """
        Note: the check 'if isinstance(event, QEvent.QLeaveEvent)' kind of fails.
        However, checking for the internal event value works, see: https://doc.qt.io/qt-6/qevent.html
        """
        if event.type() == 11:
            if not self.is_active:
                self.set_default_icon()
            return super().leaveEvent(event)



class TextToolsButton(QPushButton):
    def __init__(
            self,
            text: str,
            width: int = 32,
            height: int = 32,
            object_name: str = "TextToolsButton",
            parent=None):

        super().__init__()
        self.parent = parent
        self.setObjectName(object_name)
        self.width = width
        self.height = height
        self.setText(text)

        self.setFixedHeight(self.height)
        self.setFixedWidth(self.width)