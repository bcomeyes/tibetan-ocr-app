import pyewts
from uuid import UUID
from BDRC.Data import Encoding
from typing import List

from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QSize, QEvent, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPen,
    QPixmap,
    QResizeEvent,
    QPainter,
    QImage,
    QPainterPath,
    QFont,
    QIntValidator
)

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QScrollBar,
    QWidget,
    QLabel,
    QSpacerItem,
    QListWidget,
    QLineEdit,
    QListWidgetItem,
    QLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QFrame,
    QListView, QPushButton
)

from BDRC.Data import OCRData, OCRModel
from BDRC.Utils import get_filename
from BDRC.MVVM.viewmodel import DataViewModel, SettingsViewModel
from BDRC.Widgets.Buttons import MenuButton, TextToolsButton
from BDRC.Widgets.Dialogs import ImportFilesDialog
from BDRC.Widgets.GraphicItems import ImagePreview



class HeaderTools(QFrame):

    def __init__(self, data_view: DataViewModel, settings_view: SettingsViewModel, icon_size: int = 48):
        super().__init__()
        self.setObjectName("HeaderTools")
        self.data_view = data_view
        self.settings_view = settings_view
        self.toolbox = ToolBox(ocr_models=self.settings_view.get_ocr_models(), icon_size=icon_size)
        self.page_switcher = PageSwitcher(icon_size=icon_size)

        # bind signals
        self.data_view.dataSelected.connect(self.set_page_index)
        self.settings_view.ocrModelsChanged.connect(self.update_ocr_models)

        # build layout
        self.spacer = QLabel()
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.toolbox)
        self.layout.addWidget(self.page_switcher)
        self.layout.addWidget(self.spacer)

        self.layout.setContentsMargins(0, 10, 10, 10)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setLayout(self.layout)

    def update_page_count(self, amount: int):
        self.page_switcher.max_pages = amount

    def set_page_index(self, data: OCRData):
        _index = self.data_view.get_data_index(data.guid)
        self.page_switcher.update_page(_index)


    def update_ocr_models(self):
        _ocr_models = self.settings_view.get_ocr_models()
        self.toolbox.update_ocr_models(_ocr_models)

class ToolBox(QWidget):
    sign_new = Signal()
    sign_import_files = Signal(list)
    sign_save = Signal()
    sign_run = Signal()
    sign_run_all = Signal()
    sign_settings = Signal()
    sign_update_page = Signal(int)
    sign_on_select_model = Signal(OCRModel)

    def __init__(self, ocr_models: List[OCRModel] | None, icon_size: int = 64):
        super().__init__()
        self.setObjectName("ToolBox")
        self.ocr_models = ocr_models

        self.setFixedHeight(74)
        self.setMinimumWidth(720)
        self.icon_size = icon_size

        self.new_btn_icon = "Assets/Textures/new_light.png"
        self.import_btn_icon = "Assets/Textures/import.png"
        self.save_btn_icon = "Assets/Textures/save-disc.png"
        self.run_btn_icon = "Assets/Textures/play_btn.png"
        self.run_all_btn_icon = "Assets/Textures/play_all_btn.png"
        self.settings_btn_icon = "Assets/Textures/settings.png"

        self.btn_new = MenuButton(
            "New Project",
            self.new_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_import = MenuButton(
            "Import",
            self.import_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_save = MenuButton(
            "Save",
            self.save_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_run = MenuButton(
            "Run OCR",
            self.run_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_run_all = MenuButton(
            "Run OCR on all images",
            self.run_all_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_settings = MenuButton(
            "Settings",
            self.settings_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        #spacer
        self.spacer = QLabel("")
        self.spacer.setFixedWidth(20)

        # model selection
        self.model_selection = QComboBox()
        self.model_selection.setObjectName("ModelSelection")
        self.model_selection.setContentsMargins(80, 0, 0, 0)


        self.model_selection.setStyleSheet("""
            QListView {
                color:white;
                background-color: #172832;
                min-width: 150px;      
        }  
        """)

        if self.ocr_models is not None and len(self.ocr_models) > 0:
            for model in self.ocr_models:
                self.model_selection.addItem(model.name)

        # self.model_selection.activated.connect(self.on_select_ocr_model)
        self.model_selection.currentIndexChanged.connect(self.on_select_ocr_model)

        # build layout
        self.layout = QHBoxLayout()
        self.layout.setSizeConstraint(
           QLayout.SizeConstraint.SetMinimumSize)

        #self.layout.setSpacing(14)
        # hook up button clicks
        self.btn_new.clicked.connect(self.new)
        self.btn_import.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save)
        self.btn_run.clicked.connect(self.run)
        self.btn_run_all.clicked.connect(self.run_all)
        self.btn_settings.clicked.connect(self.settings)

        # build layout
        self.layout.addWidget(self.btn_new)
        self.layout.addWidget(self.btn_import)
        self.layout.addWidget(self.btn_save)
        self.layout.addWidget(self.btn_run)
        self.layout.addWidget(self.btn_run_all)
        self.layout.addWidget(self.btn_settings)
        self.layout.addWidget(self.spacer)
        self.layout.addWidget(self.model_selection)

        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.setContentsMargins(10, 0, 0, 0)
        self.setLayout(self.layout)


    def new(self):
        self.sign_new.emit()

    def load_image(self):
        dialog = ImportFilesDialog(self)
        if dialog.exec():
            _file_paths = dialog.selectedFiles()
            if _file_paths and len(_file_paths) > 0:
                self.sign_import_files.emit(_file_paths)

    def save(self):
        self.sign_save.emit()

    def run(self):
        self.sign_run.emit()

    def run_all(self):
        self.sign_run_all.emit()

    def settings(self):
        self.sign_settings.emit()

    def update_page(self, index: int):
        self.sign_update_page.emit(index)

    def on_select_ocr_model(self, index: int):
        self.sign_on_select_model.emit(self.ocr_models[index])

    def update_ocr_models(self, ocr_models: List[OCRModel]):
        self.ocr_models = ocr_models

        if self.ocr_models is not None and len(self.ocr_models) > 0:
            for model in self.ocr_models:
                self.model_selection.addItem(model.name)


class PageSwitcher(QWidget):
    sign_on_page_changed = Signal(int)

    def __init__(self, pages: int = 0, icon_size: int = 40):
        super().__init__()
        self.setObjectName("PageSwitcher")
        self.icon_size = icon_size
        self.setFixedHeight(74)
        self.setMaximumSize(264, 74)
        self.setContentsMargins(0, 0, 0, 0)
        self.max_pages = pages
        self.current_index = 0
        self.layout = QHBoxLayout()

        self.current_page = QLineEdit()
        self.current_page.setStyleSheet("""
            color: #ffffff;
            font-weight: bold;
            background: #434343;
        """)

        self.current_page.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.current_page.setObjectName("PageSelector")
        self.current_page.setFixedSize(100, 40)

        self.prev_btn_icon = "Assets/Textures/prev.png"
        self.next_btn_icon = "Assets/Textures/next.png"

        self.prev_button = MenuButton(
            "Previous image",
            self.prev_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.next_button = MenuButton(
            "Next image",
            self.next_btn_icon,
            width=self.icon_size,
            height=self.icon_size,
        )

        # build layout
        self.layout.addWidget(self.prev_button)
        self.layout.addWidget(self.current_page)
        self.layout.addWidget(self.next_button)
        self.setLayout(self.layout)

        self.prev_button.clicked.connect(self.prev)
        self.next_button.clicked.connect(self.next)


    def update_page(self, index: int):
        self.current_index = index
        self.current_page.setText(str(self.current_index+1))

    def prev(self):
        next_index = self.current_index - 1

        if next_index >= 0:
            self.update_page(next_index)
            self.sign_on_page_changed.emit(next_index)

    def next(self):
        next_index = self.current_index + 1

        if not next_index > self.max_pages-1:
            self.update_page(next_index)
            self.sign_on_page_changed.emit(next_index)


class PTGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setScene(self.scene)
        self.setMinimumHeight(200)

        self.zoom_in_factor = 0.8
        self.zoom_clamp = False
        self.default_zoom_step = 10
        self.current_zoom_step = 10
        self.zoom_range = [0, 20]

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        self.default_scrollbar_policy = Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        self.setHorizontalScrollBarPolicy(self.default_scrollbar_policy)
        self.setVerticalScrollBarPolicy(self.default_scrollbar_policy)

        self.verticalScrollBar().setSliderPosition(1)
        self.horizontalScrollBar().setSliderPosition(1)

        self.v_scrollbar = QScrollBar(self)
        self.v_scrollbar.setObjectName("VerticalScrollBar")
        self.h_scrollbar = QScrollBar(self)
        self.v_scrollbar.setObjectName("HorizontalScrollbar")

        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)

        self.v_scrollbar.setStyleSheet(
            """

            QScrollBar:vertical {
                border: none;
                background: #2d2d46;
                width: 25px;
                margin: 10px 5px 15px 10px;
                border-radius: 0px;
             }

            QScrollBar::handle:vertical {	
                border: 2px solid #A40021;
                background-color: #A40021;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover{	
                background-color: #C80021;
            }
            QScrollBar::handle:vertical:pressed {	
                background-color: #C80021;
            }

            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-line:vertical {
                height: 0px;
            }

            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            """
        )

        self.h_scrollbar.setStyleSheet(
            """
            QScrollBar:horizontal {
                border: none;
                background: #2d2d46;
                height: 30px;
                margin: 10px 10px 10px 10px;
                border-radius: 0px;
            }
            QScrollBar::handle:horizontal {
                border: 2px solid #A40021;
                background-color: #A40021;
                min-width: 30px;
                border-radius: 3px;
            }
            QScrollBar::add-line:horizontal {
                width: 0px;
            }
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal
            {
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
            {
                background: none;
            }
        """
        )

    def enable_rubberband(self):
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    def disable_rubberband(self):
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def hide_scrollbars(self):
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def show_scrollbars(self):
        self.setHorizontalScrollBarPolicy(self.default_scrollbar_policy)
        self.setVerticalScrollBarPolicy(self.default_scrollbar_policy)

    """
    def viewportEvent(self, event):
        super().viewportEvent(event)
        print(f"ViewPortEvent => {event.type()}")

        return True
    """
    def wheelEvent(self, event):
        if event.source() == Qt.MouseEventSource.MouseEventSynthesizedBySystem:
            self.handle_touch_zoom(event.angleDelta().y())
        else:
            self.handle_mouse_zoom(event.angleDelta().y())

    def handle_touch_zoom(self, y_delta: int):
        if y_delta > 6:
            if self.zoom_range[0] <= self.current_zoom_step < self.zoom_range[-1]:
                zoom_factor = 0.99
                self.current_zoom_step += 0.1

                if self.current_zoom_step > self.zoom_range[-1]:
                    self.current_zoom_step = self.zoom_range[-1]
                    return
                self.scale(zoom_factor, zoom_factor)

        elif y_delta < - 6:
            if self.zoom_range[0] < self.current_zoom_step <= self.zoom_range[-1]:
                zoom_factor = 1.01
                self.current_zoom_step -= 0.1
                if self.current_zoom_step < self.zoom_range[0]:
                    self.current_zoom_step = self.zoom_range[0]
                    return
                self.scale(zoom_factor, zoom_factor)

    def handle_mouse_zoom(self, y_delta: int):
        if y_delta > 0:
            if self.zoom_range[0] <= self.current_zoom_step < self.zoom_range[-1]:
                zoom_factor = self.zoom_in_factor
                self.current_zoom_step += 1
                self.scale(zoom_factor, zoom_factor)

        else:
            if self.zoom_range[0] < self.current_zoom_step <= self.zoom_range[-1]:
                zoom_factor = 1 / self.zoom_in_factor
                self.current_zoom_step -= 1
                self.scale(zoom_factor, zoom_factor)

    def reset_scaling(self):
        self.resetTransform()
        self.current_zoom_step = self.default_zoom_step

    def fit_in_view(self, brect: QRectF):
        if not self.default_zoom_step < self.current_zoom_step < self.default_zoom_step:
            _target_zoom_step = self.default_zoom_step - self.current_zoom_step

        else:
            return

        _zoom_factor = self.zoom_in_factor ** _target_zoom_step
        self.scale(_zoom_factor, _zoom_factor)
        self.current_zoom_step = 10


class PTGraphicsScene(QGraphicsScene):
    # see: https://forum.qt.io/topic/101616/pyside2-qtcore-signal-object-has-no-attribute-emit/4
    left_click_signal = Signal(QPoint)
    left_release_signal = Signal(QPoint)
    right_click_signal = Signal(QPoint)
    right_release_signal = Signal(QPoint)

    clear_selections = Signal()

    def __init__(self, scene, width: int = 2000, height: int = 2000, parent=None):
        super().__init__(parent)
        self._scene = scene
        self._scene_width = width
        self._scene_height = height
        self._line_color = QColor("#2f2f2f")
        self._background_color = QColor("#393939")
        self._pen_light = QPen(self._line_color)
        self._pen_light.setWidth(10)
        self._bg_image = QPixmap("Assets/Themes/Dark/background_grid.jpg")

        self.set_scene(self._scene_width, self._scene_height)
        self.setBackgroundBrush(self._bg_image)

    def set_scene(self, width: int, height: int) -> None:
        self.setSceneRect(0, 0, width, height)

    def add_item(self, item: QGraphicsItem, z_order: int):
        _brect = item.boundingRect()
        x = _brect.left()
        y = _brect.top()
        width = _brect.width()
        height = _brect.height()

        self.setSceneRect(x, y, width, height)
        item.setZValue(z_order)
        self.addItem(item)

    def remove_item(self, item: QGraphicsItem):
        self.removeItem(item)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(e)

    def get_current_item_pos(self) -> QPointF:
        canvas_items = self.items(order=Qt.SortOrder.AscendingOrder)

        for _item in canvas_items:
            if isinstance(_item, ImagePreview):
                return _item.scenePos()
            else:
                return QPointF(0, 0)

        return QPointF(0, 0)


class Canvas(QFrame):
    def __init__(self, width: int = 2000, height: int = 2000):
        super().__init__()

        self.default_width = width
        self.default_height = height
        self.setMinimumHeight(200)

        self.current_width = self.default_width
        self.current_height = self.default_height
        self.current_item_pos = QPointF(0.0, 0.0)

        self.gr_scene = PTGraphicsScene(
            self, width=self.current_width, height=self.current_height
        )
        self.view = PTGraphicsView(self.gr_scene)
        self.view.setScene(self.gr_scene)

        self.toggle_prev_btn_icon = "Assets/Textures/toggle_prev.png"
        self.fit_view_icon = "Assets/Textures/fit_to_canvas.png"
        self.zoom_in_icon = "Assets/Textures/plus_sign.png"
        self.zoom_out_icon = "Assets/Textures/minus_sign.png"

        self.toggle_prev_btn = MenuButton(
            "Toggle line preview",
            self.toggle_prev_btn_icon,
            width=20,
            height=20
        )

        self.fit_in_btn = MenuButton(
            "Fit image in view",
            self.fit_view_icon,
            width=20,
            height=20
        )


        self.zoom_in_btn = MenuButton(
            "Zoom in",
            self.zoom_in_icon,
            width=20,
            height=20
        )

        self.zoom_out_btn = MenuButton(
            "Zoom out",
            self.zoom_out_icon,
            width=20,
            height=20
        )

        # bind signals
        self.toggle_prev_btn.clicked.connect(self.handle_preview_toggle)
        self.fit_in_btn.clicked.connect(self.fit_in_view)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        self.canvas_tools_layout = QHBoxLayout()
        self.canvas_tools_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.canvas_tools_layout.addWidget(self.toggle_prev_btn)
        self.canvas_tools_layout.addWidget(self.fit_in_btn)
        self.canvas_tools_layout.addWidget(self.zoom_in_btn)
        self.canvas_tools_layout.addWidget(self.zoom_out_btn)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.canvas_tools_layout)
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

        self.setStyleSheet(
            """
                    color: #ffffff;
                    background-color: #100F0F;
                    border: 2px solid #100F0F; 
                    border-radius: 8px;
                """
        )

    def update_display_position(self, position: QPointF):
        print(f"Canvas: updating tracked item position: {position}")
        self.current_item_pos = position

    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            _new_size = event.size()
            self.current_width = _new_size.width()
            self.current_height = _new_size.height()
            self.gr_scene.setSceneRect(
                QRectF(0, 0, self.current_width, self.current_height)
            )

    def set_preview(self, data: OCRData):
        self.view.reset_scaling()
        self.gr_scene.clear()

        preview_item = ImagePreview(data.image_path, data.lines, data.angle)
        brect = preview_item.boundingRect()
        _pos = QPointF(0, 0)
        preview_item.setPos(_pos)

        self.gr_scene.add_item(preview_item, 1)
        self.view.fitInView(brect, Qt.AspectRatioMode.KeepAspectRatio)

    def handle_preview_toggle(self):
        for item in self.gr_scene.items():
            if isinstance(item, ImagePreview):
                if item.is_in_preview:
                    item.show_image()
                else:
                    item.show_preview()

    def fit_in_view(self):
        for item in self.gr_scene.items():
            if isinstance(item, ImagePreview):
                b_rect = item.boundingRect()
                item.setPos(0, 0)
                self.view.fit_in_view(b_rect)


    def zoom_in(self):
        self.view.handle_mouse_zoom(-1)

    def zoom_out(self):
        self.view.handle_mouse_zoom(1)


class ImageList(QListWidget):
    sign_on_selected_item = Signal(UUID)
    """
    https://stackoverflow.com/questions/64576846/how-to-paint-an-outline-when-hovering-over-a-qlistwidget-item
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setObjectName("ImageGalleryList")
        self.setObjectName("ImageGalleryItem")
        self.setFlow(QListView.Flow.TopToBottom)
        self.setMouseTracking(True)
        self.itemClicked.connect(self.on_item_clicked)

        self.v_scrollbar = QScrollBar(self)
        self.h_scrollbar = QScrollBar(self)
        self.v_scrollbar.setStyleSheet(
            """
            
            QScrollBar:vertical {
                border: none;
                background: #2d2d46;
                width: 25px;
                margin: 10px 5px 15px 10px;
                border-radius: 0px;
             }
            
            QScrollBar::handle:vertical {	
                border: 2px solid #A40021;
                background-color: #A40021;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover{	
                background-color: #C80021;
            }
            QScrollBar::handle:vertical:pressed {	
                background-color: #C80021;
            }
            
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
           
            QScrollBar::add-line:vertical {
                height: 0px;
            }
            
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            """
        )

        self.h_scrollbar.setStyleSheet(
            """
            QScrollBar:horizontal {
                border: none;
                background: #2d2d46;
                height: 30px;
                margin: 10px 10px 10px 10px;
                border-radius: 0px;
            }
            QScrollBar::handle:horizontal {
                border: 2px solid #A40021;
                background-color: #A40021;
                min-width: 30px;
                border-radius: 3px;
            }
            QScrollBar::add-line:horizontal {
                width: 0px;
            }
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal
            {
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
            {
                background: none;
            }
        """
        )

        self.setAutoScrollMargin(20)
        # self.scroll_bar.setObjectName("PalmTreeBar")

        # setting vertical scroll bar to it
        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)
        # self.itemEntered.connect(self.on_hover_item)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet(
            """
                background-color: #100f0f;
            """
        )

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(item)  # returns an instance of CanvasHierarchyEntry

        if isinstance(_list_item_widget, ImageListWidget):
            self.sign_on_selected_item.emit(_list_item_widget.guid)

    def mouseMoveEvent(self, e):
        item = self.itemAt(e.pos())
        if item is not None:
            print("ImageList->ITEM")
            _list_item_widget = self.itemWidget(item)
            print(f"Item Hovered over: {_list_item_widget}")

        """if isinstance(_list_item_widget, ImageListWidget):
            _list_item_widget.is_hovered = True
        """

    def event(self, event):
        if event.type() == QEvent.Type.HoverEnter:
            #print("QFrame->enter")
            pass
        elif event.type() == QEvent.Type.HoverLeave:
            #print("QFrame->leave")
            pass

        return super().event(event)


class ImageThumb(QFrame):
    def __init__(self, image_path: str, max_height: int = 140):
        super().__init__()
        # TODO: Setting this does actually not work
        self.image_path = image_path
        self.setFixedHeight(max_height)
        self.setMinimumWidth(220)
        self.max_height = 140
        self.current_width = 220
        self.round_rect_margin = 6
        self.round_rect_radius = 14
        self.qimage = QImage(self.image_path).scaledToHeight(max_height)
        self.pixmap = QPixmap(image_path)
        self.brush = QBrush(self.pixmap)

        self._pen_hover = QPen(QColor("#fce08d"))
        self._pen_hover.setWidth(6)

        self._pen_select = QPen(QColor("#ffad00"))
        self._pen_select.setWidth(6)

        self.source_img = QImage(self.current_width, 140, QImage.Format.Format_ARGB32)
        self.source_img.fill(Qt.GlobalColor.blue)

        self.dest_img = QImage(self.current_width, 140, QImage.Format.Format_ARGB32)
        self.dest_img.fill(Qt.GlobalColor.transparent)

        self.clip_path = QPainterPath()
        self.clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin,
                self.round_rect_margin,
                -self.round_rect_margin,
                -self.round_rect_margin,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.hover_clip_path = QPainterPath()
        self.hover_clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin - 2,
                self.round_rect_margin - 2,
                -self.round_rect_margin + 2,
                -self.round_rect_margin + 2,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.is_hovered = False
        self.is_selected = False

    def resize_thumb(self, new_width: int):
        self.current_width = new_width
        self.source_img = QImage(
            self.current_width, self.max_height, QImage.Format.Format_ARGB32
        )
        self.source_img.fill(Qt.GlobalColor.blue)

        self.dest_img = QImage(
            self.current_width, self.max_height, QImage.Format.Format_ARGB32
        )
        self.dest_img.fill(Qt.GlobalColor.transparent)

        self.clip_path = QPainterPath()

        self.clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin,
                self.round_rect_margin,
                -self.round_rect_margin,
                -self.round_rect_margin,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.hover_clip_path = QPainterPath()
        self.hover_clip_path.addRoundedRect(
            self.source_img.rect().adjusted(
                self.round_rect_margin - 2,
                self.round_rect_margin - 2,
                -self.round_rect_margin + 2,
                -self.round_rect_margin + 2,
            ),
            self.round_rect_radius,
            self.round_rect_radius,
        )

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipPath(self.clip_path)
        painter.drawImage(0, 0, self.qimage)

        if self.is_hovered:
            painter.setClipPath(self.hover_clip_path)
            painter.drawImage(0, 0, self.qimage)
            path_outline = QPainterPath()
            path_outline.addRoundedRect(
                self.round_rect_margin,
                self.round_rect_margin,
                self.current_width - 2 * self.round_rect_margin,
                self.max_height - 2 * self.round_rect_margin,
                self.round_rect_radius,
                self.round_rect_radius,
            )

            painter.setPen(self._pen_hover)
            painter.drawPath(path_outline.simplified())

        elif self.is_selected:
            painter.setClipPath(self.hover_clip_path)
            painter.drawImage(0, 0, self.qimage)
            path_outline = QPainterPath()
            path_outline.addRoundedRect(
                self.round_rect_margin,
                self.round_rect_margin,
                self.current_width - 2 * self.round_rect_margin,
                self.max_height - 2 * self.round_rect_margin,
                self.round_rect_radius,
                self.round_rect_radius,
            )

            painter.setPen(self._pen_select)
            painter.drawPath(path_outline.simplified())

        else:
            painter.setClipPath(self.clip_path)
            painter.drawImage(0, 0, self.qimage)

        painter.end()


class ImageListWidget(QWidget):
    def __init__(self, guid: UUID, image_path: str, width: int, height: int):
        super().__init__()
        self.guid = guid
        self.image_path = image_path
        self.file_name = get_filename(image_path)
        self.base_width = width
        self.base_height = height
        self.thumb = ImageThumb(image_path)
        self.label = QLabel()
        self.label.setContentsMargins(0, 0, 0, 0)
        self.label.setText(self.file_name)
        self.setBaseSize(self.base_width, self.base_height)

        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)

        self.layout.addWidget(self.thumb)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.is_active = False

        self.label.setStyleSheet("""
            color: #ffffff;

        """)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            self.thumb.resize_thumb(event.size().width()-20)
            # TODO: scale down the entire List..?
            #self.resize()

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            self.thumb.is_hovered = True
            self.thumb.is_selected = False
            self.thumb.update()

        elif event.type() == QEvent.Type.Leave:
            self.thumb.is_hovered = False

            if self.is_active:
                self.thumb.is_selected = True
            else:
                self.thumb.is_selected = False

            self.thumb.update()

        return super().event(event)

    def select(self):
        self.is_active = True
        self.thumb.is_selected = True
        self.thumb.is_hovered = False
        self.thumb.update()

    def unselect(self):
        self.is_active = False
        self.thumb.is_selected = False
        self.thumb.is_hovered = False
        self.thumb.update()



class ImageGallery(QFrame):
    def __init__(self, viewmodel: DataViewModel):
        super().__init__()
        self.view_model = viewmodel
        self.setObjectName("ImageGallery")
        self.setMinimumHeight(600)
        self.setMaximumWidth(456)
        self.setContentsMargins(20, 20, 20, 20)
        self.setContentsMargins(0, 0, 0, 0)

        # build layout
        self.image_label = QLabel(self)
        self.image_label.setContentsMargins(6, 0, 0, 0)
        self.image_pixmap = QPixmap("Assets/Textures/BDRC_Logo.png").scaled(
            QSize(140, 90), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(self.image_pixmap)

        self.layout = QVBoxLayout()
        self.spacer = QSpacerItem(320, 10)
        self.image_list = ImageList(self)

        self.layout.addWidget(self.image_label)
        self.layout.addItem(self.spacer)
        self.layout.addWidget(self.image_list)
        self.setLayout(self.layout)

        self.image_list.setStyleSheet(
            """
            background-color: #100f0f;
            border: 4px solid #100f0f;
                       
            QListWidget {
                color: #ffffff;
                background-color: #100f0f; 
                border: 4px solid #100f0f;    
            }
            
            QListWidget::item:selected {
                background: #2d2d46;
            }
            """
        )

        self.current_width = 320

        # connect signals
        self.view_model.dataChanged.connect(self.add_data)
        self.view_model.dataCleared.connect(self.clear_data)
        self.view_model.dataAutoSelected.connect(self.focus_page)
        self.image_list.sign_on_selected_item.connect(self.handle_item_selection)

    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            _new_size = event.size()
            self.current_width = _new_size.width()

    def handle_item_selection(self, guid: UUID):
        for idx in range(self.image_list.count()):
                item = self.image_list.item(idx)

                item_widget = self.image_list.itemWidget(item)
                if isinstance(item_widget, ImageListWidget):
                    if item_widget.guid == guid:
                        item_widget.select()
                    else:
                        item_widget.unselect()

        self.view_model.select_data_by_guid(guid)

    def select_page(self, index: int):
        for idx in range(self.image_list.count()):
            if idx == index:
                item = self.image_list.item(idx)
                item.setSelected(True)
                item_widget = self.image_list.itemWidget(item)

                if isinstance(item_widget, ImageListWidget):
                    item_widget.select()
                    self.view_model.select_data_by_guid(item_widget.guid)
            else:
                item = self.image_list.item(idx)
                item.setSelected(False)
                item_widget = self.image_list.itemWidget(item)

                if isinstance(item_widget, ImageListWidget):
                    item_widget.unselect()

    def focus_page(self, data: OCRData):
        for idx in range(self.image_list.count()):
            item = self.image_list.item(idx)
            item_widget = self.image_list.itemWidget(item)

            if isinstance(item_widget, ImageListWidget):
                if item_widget.guid == data.guid:
                    item.setSelected(True)  # what does this built-in method actually do..?
                    item_widget.select()
                    self.image_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                else:
                    item.setSelected(False)
                    item_widget.unselect()


    def add_data(self, data: list[OCRData]):
        _sizeHint = self.sizeHint()
        _targetWidth = _sizeHint.width()-80

        for _data in data:
            image_item = QListWidgetItem()
            image_item.setSizeHint(QSize(_targetWidth, 200))
            image_widget = ImageListWidget(
                _data.guid,
                _data.image_path,
                width=_targetWidth,
                height=200
            )

            self.image_list.addItem(image_item)
            self.image_list.setItemWidget(image_item, image_widget)

    def clear_data(self):
        self.image_list.clear()



class TextWidgetList(QListWidget):
    sign_on_selected_item = Signal(UUID)

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setObjectName("TextListWidget")
        self.setFlow(QListView.Flow.TopToBottom)
        self.setMouseTracking(True)
        self.itemClicked.connect(self.on_item_clicked)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # self.v_scrollbar = QScrollBar(self)
        self.v_scrollbar = QScrollBar(self)
        self.h_scrollbar = QScrollBar(self)
        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)

        self.v_scrollbar.setStyleSheet(
            """
            QScrollBar:vertical {
                border: none;
                background: #2d2d46;
                width: 25px;
                margin: 10px 5px 15px 10px;
                border-radius: 0px;
             }

            QScrollBar::handle:vertical {	
                border: 2px solid #A40021;
                background-color: #A40021;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover{	
                background-color: #C80021;
            }
            QScrollBar::handle:vertical:pressed {	
                background-color: #C80021;
            }

            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-line:vertical {
                height: 0px;
            }

            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }

            """
        )

        self.h_scrollbar.setStyleSheet(
            """
            QScrollBar:horizontal {
                border: none;
                background: #2d2d46;
                height: 30px;
                margin: 10px 10px 10px 10px;
                border-radius: 0px;
            }
            QScrollBar::handle:horizontal {
                border: 2px solid #A40021;
                background-color: #A40021;
                min-width: 30px;
                border-radius: 3px;
            }
            QScrollBar::add-line:horizontal {
                width: 0px;
            }
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal
            {
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
            {
                background: none;
            }
        """
        )

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(item)  # returns an instance of CanvasHierarchyEntry

        if isinstance(_list_item_widget, TextListWidget):
            """
            TODO: enable highlighting of the selected text line in the canvas preview
            """
            pass
            #print("TextWidgetList -> selected Text Widget")
            # self.sign_on_selected_item.emit(_list_item_widget.guid)

    def mouseMoveEvent(self, e):
        item = self.itemAt(e.pos())
        if item is not None:
            print("TextWidgetList->ITEM")
            _list_item_widget = self.itemWidget(item)

        """if isinstance(_list_item_widget, ImageListWidget):
            _list_item_widget.is_hovered = True
        """

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            #print("QFrame->enter")
            pass
        elif event.type() == QEvent.Type.Leave:
            #print("QFrame->leave")
            pass

        return super().event(event)


class TextListWidget(QWidget):
    """
    Custom widget holding the actual text data
    """

    def __init__(self, text: str, font: str, font_size: int = 24):
        super().__init__()
        self.text = text
        self.font = font
        self.font_size = font_size
        self.label = QLabel()
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setFont(QFont(self.font, font_size))
        self.label.setText(self.text)
        self.is_hovered = False
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)


        label_size = self.label.sizeHint()
        self.setFixedHeight(label_size.height() + 24) # that is a bit hacky, is it possible to inferr the size of the label based on the rendered text..?

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            self.is_hovered = True

        elif event.type() == QEvent.Type.Leave:
            self.is_hovered = False

        return super().event(event)

    def update_text(self, text):
        self.text = text


class TextView(QFrame):
    def __init__(self, font_size: int = 14, encoding: Encoding = Encoding.Unicode):
        super().__init__()
        self.setObjectName("TextView")
        self.font_size = font_size
        self.encoding = encoding
        self.default_font = "Assets/Fonts/Monlam/Monlam-bodyig Regular.ttf"
        self.current_font = self.default_font
        self.converter = pyewts.pyewts()
        self.setContentsMargins(10, 0, 10, 0)
        self.text_lines = []
        #self.setMinimumHeight(80)
        #self.setMinimumWidth(600)

        self.text_widget_list = TextWidgetList()

        self.zoom_in_btn = TextToolsButton("+")
        self.zoom_in_btn.setStyleSheet("""
            background-color: #3f3f3f;
            border: 2px solid #1d1d1d;
            border-radius: 4px;
        """)

        self.zoom_out_btn = TextToolsButton("-")
        self.zoom_out_btn.setStyleSheet("""
                    background-color: #3f3f3f;
                    border: 2px solid #1d1d1d;
                    border-radius: 4px;
                """)


        self.spacer = QLabel()

        # bind signals
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        # build layout
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.zoom_in_btn)
        self.button_layout.addWidget(self.zoom_out_btn)
        self.button_layout.addWidget(self.spacer)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.text_widget_list)
        self.setLayout(self.layout)
        self.setStyleSheet(
            """
                color: #ffffff;
                background-color: #100F0F;
                border: 2px solid #100F0F; 
                border-radius: 4px;
            """
        )

    def zoom_in(self):
        if len(self.text_lines) == 0:
            return

        self.text_widget_list.clear()

        for text_line in self.text_lines:
            text_line = self.converter.toUnicode(text_line)
            list_item = QListWidgetItem()

            self.font_size = self.font_size+1
            text_widget = TextListWidget(text_line, self.current_font, self.font_size)
            text_size = text_widget.sizeHint()

            if text_size.width() < 800:
                list_item.setSizeHint(QSize(800, text_size.height()))
            else:
                list_item.setSizeHint(QSize(text_size.width(), text_size.height()))

            self.text_widget_list.addItem(list_item)
            self.text_widget_list.setItemWidget(list_item, text_widget)

        for idx in range(self.text_widget_list.count()):
            if idx % 2 == 0:
                _qBrush = QBrush(QColor("#172832"))
            else:
                _qBrush = QBrush(QColor("#1d1c1c"))
            self.text_widget_list.item(idx).setBackground(_qBrush)

    def zoom_out(self):
        if len(self.text_lines) == 0:
            return

        self.text_widget_list.clear()

        for text_line in self.text_lines:
            text_line = self.converter.toUnicode(text_line)
            list_item = QListWidgetItem()
            self.font_size = self.font_size - 1

            text_widget = TextListWidget(text_line, self.current_font, self.font_size)
            text_size = text_widget.sizeHint()

            if text_size.width() < 800:
                list_item.setSizeHint(QSize(800, text_size.height()))
            else:
                list_item.setSizeHint(QSize(text_size.width(), text_size.height()))

            self.text_widget_list.addItem(list_item)
            self.text_widget_list.setItemWidget(list_item, text_widget)

        for idx in range(self.text_widget_list.count()):
            if idx % 2 == 0:
                _qBrush = QBrush(QColor("#172832"))
            else:
                _qBrush = QBrush(QColor("#1d1c1c"))
            self.text_widget_list.item(idx).setBackground(_qBrush)

    def update_text(self, text_lines: List[str]):
        self.text_lines = text_lines
        self.text_widget_list.clear()

        for text_line in text_lines:
            text_line = self.converter.toUnicode(text_line)
            list_item = QListWidgetItem()
            text_widget = TextListWidget(text_line, self.current_font, self.font_size)

            text_size = text_widget.sizeHint()

            if text_size.width() < 800:
                list_item.setSizeHint(QSize(800, text_size.height()))
            else:
                list_item.setSizeHint(QSize(text_size.width(), text_size.height()))

            self.text_widget_list.addItem(list_item)
            self.text_widget_list.setItemWidget(list_item, text_widget)

        for idx in range(self.text_widget_list.count()):
            if idx % 2 == 0:
                _qBrush = QBrush(QColor("#172832"))
            else:
                _qBrush = QBrush(QColor("#1d1c1c"))
            self.text_widget_list.item(idx).setBackground(_qBrush)

    def update_font(self, font_path: str):
        self.current_font = font_path

    def update_font_size(self, font_size: int):
        self.font_size = font_size
