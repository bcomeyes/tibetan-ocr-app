import os
import pyewts
from uuid import UUID
from BudaOCR import Config
from BudaOCR.Data import Encoding

from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QSize, QEvent, QRectF
from PySide6.QtGui import (
    QIcon,
    QBrush,
    QColor,
    QPen,
    QPixmap,
    QResizeEvent,
    QPainter,
    QImage,
    QPainterPath,
    QFont,
    QIntValidator,
)

from PySide6.QtWidgets import (
    QComboBox,
    QScrollBar,
    QWidget,
    QLabel,
    QSpacerItem,
    QListWidget,
    QLineEdit,
    QListWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QFrame,
    QListView
)

from BudaOCR.Data import BudaOCRData
from BudaOCR.Controller import ImageController
from BudaOCR.Utils import get_filename
from BudaOCR.MVVM.viewmodel import BudaViewModel
from BudaOCR.Widgets.Buttons import HeaderButton
from BudaOCR.Widgets.Dialogs import ImportFilesDialog
from BudaOCR.Widgets.GraphicItems import ImagePreview


class HeaderTools(QWidget):

    def __init__(self, icon_size: int = 48):
        super().__init__()

        self.toolbox = ToolBox(ocr_models=["Modern", "Woodblock"], icon_size=icon_size)
        self.page_switcher = PageSwitcher(icon_size=icon_size)

        # build layout
        self.spacer = QLabel()
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.toolbox)
        self.layout.addWidget(self.page_switcher)
        #self.layout.addWidget(self.ocr_tools)
        self.layout.addWidget(self.spacer)
        self.setLayout(self.layout)

    def update_page_count(self, amount: int):
        self.page_switcher.max_pages = amount

    def set_page_index(self, index: int):
        self.page_switcher.update_page(index)


class ToolBox(QFrame):
    sign_new = Signal()
    sign_import_files = Signal(list)
    sign_save = Signal()
    sign_run = Signal()
    sign_run_all = Signal()
    sign_settings = Signal()
    sign_update_page = Signal(int)
    sign_on_select_model = Signal(str)

    def __init__(self, ocr_models: list[str], icon_size: int = 64):
        super().__init__()
        self.ocr_models = ocr_models

        self.setFixedWidth(460)
        self.setFixedHeight(74)
        self.setContentsMargins(0, 0, 0, 0)
        self.icon_size = icon_size

        self.new_btn_icon = QIcon("Assets/Themes/Dark/new_light.png")
        self.new_btn_icon_hover = QIcon("Assets/Themes/Dark/new_hover.png")

        self.import_btn_icon = QIcon("Assets/Themes/Dark/import.png")
        self.import_btn_icon_hover = QIcon("Assets/Themes/Dark/import_hover.png")

        self.save_btn_icon = QIcon("Assets/Themes/Dark/save-disc.png")
        self.save_btn_icon_hover = QIcon("Assets/Themes/Dark/save-disc_hover.png")

        self.run_btn_icon = QIcon("Assets/Themes/Dark/play_light.png")
        self.run_btn_icon_hover = QIcon("Assets/Themes/Dark/play_light_hover.png")

        self.run_all_btn_icon = QIcon("Assets/Themes/Dark/play_all_light.png")
        self.run_all_btn_icon_hover = QIcon("Assets/Themes/Dark/play_all_light.png") # TODO: Create an Icon for that

        self.settings_btn_icon = QIcon("Assets/Themes/Dark/settings.png")
        self.settings_btn_icon_hover = QIcon("Assets/Themes/Dark/settings_hover.png")

        self.btn_new = HeaderButton(
            self.new_btn_icon,
            self.new_btn_icon_hover,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_import = HeaderButton(
            self.import_btn_icon,
            self.import_btn_icon_hover,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_save = HeaderButton(
            self.save_btn_icon,
            self.save_btn_icon_hover,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_run = HeaderButton(
            self.run_btn_icon,
            self.run_btn_icon_hover,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_run_all = HeaderButton(
            self.run_all_btn_icon,
            self.run_all_btn_icon_hover,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.btn_settings = HeaderButton(
            self.settings_btn_icon,
            self.settings_btn_icon_hover,
            width=self.icon_size,
            height=self.icon_size,
        )

        # model selection
        self.model_selection = QComboBox()
        self.model_selection.setFixedWidth(140)

        for model in self.ocr_models:
            self.model_selection.addItem(model)

        # self.model_selection.activated.connect(self.on_select_ocr_model)
        self.model_selection.currentIndexChanged.connect(self.on_select_ocr_model)

        # build layout
        self.layout = QHBoxLayout()
        self.spacer = QLabel()
        self.layout.addWidget(self.btn_new)
        self.layout.addWidget(self.btn_import)
        self.layout.addWidget(self.btn_save)
        self.layout.addWidget(self.btn_run)
        self.layout.addWidget(self.btn_run_all)
        self.layout.addWidget(self.btn_settings)
        self.layout.addWidget(self.model_selection)
        self.layout.addWidget(self.spacer)
        self.setLayout(self.layout)

        # TODO: move this to global style definition
        self.setStyleSheet(
            """
                    color: #ffffff;
                    background-color: #100F0F;
                    border: 2px solid #100F0F; 
                    border-radius: 8px;
        """
        )

        # hook up button clicks
        self.btn_new.clicked.connect(self.new)
        self.btn_import.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save)
        self.btn_run.clicked.connect(self.run)
        self.btn_run_all.clicked.connect(self.run_all)
        self.btn_settings.clicked.connect(self.settings)

    def new(self):
        self.sign_new.emit()

    def load_image(self):

        dialog = ImportFilesDialog(self)
        if dialog.exec():
            _file_paths = dialog.selectedFiles()
            if _file_paths and len(_file_paths) > 0:
                print(f"Loading {len(_file_paths)} Images")
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
        print(f"Selecting OCR Model : {index}")
        self.sign_on_select_model.emit(self.ocr_models[index])


class PageSwitcher(QFrame):
    sign_on_page_changed = Signal(int)

    def __init__(self, pages: int = 0, icon_size: int = 40):
        super().__init__()
        self.icon_size = icon_size
        self.setFixedHeight(60)
        self.setMaximumSize(240, 60)
        self.setContentsMargins(0, 0, 0, 0)
        self.max_pages = pages
        self.current_index = 0
        self.layout = QHBoxLayout()

        self.current_page = QLineEdit()
        self.current_page.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.current_page.setStyleSheet(
            """
            border: 1px solid white;
        """
        )

        self.current_page.setFixedSize(100, 40)
        self.prev_btn_icon = QIcon("Assets/Themes/Dark/prev.png")
        self.prev_btn_icon_hover = QIcon("Assets/Themes/Dark/prev_hover.png")
        self.next_btn_icon = QIcon("Assets/Themes/Dark/next.png")
        self.next_btn_icon_hover = QIcon("Assets/Themes/Dark/next_hover.png")

        self.prev_button = HeaderButton(
            self.prev_btn_icon,
            self.prev_btn_icon_hover,
            width=self.icon_size,
            height=self.icon_size,
        )

        self.next_button = HeaderButton(
            self.next_btn_icon,
            self.next_btn_icon_hover,
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

        self.setStyleSheet(
            """
            color: #ffffff;
            background-color: #100F0F;
            border: 2px solid #100F0F; 
            border-radius: 8px;
        """
        )

    def update_page(self, index: int):
        self.current_index = index
        print(f"Updating current page text: {index}")
        self.current_page.setText(str(self.current_index))

    def prev(self):
        next_index = self.current_index - 1
        if not next_index - 1 < 0:
            self.update_page(next_index)
            self.sign_on_page_changed.emit(next_index)
        else:
            print("Beginning of dataset reached")

    def next(self):
        print(f"Current Index: {self.current_index}")
        next_index = self.current_index + 1
        if not next_index > self.max_pages:
            print(f"Selecting next index: {next_index}")
            self.update_page(next_index)
            self.sign_on_page_changed.emit(next_index)
        else:
            print("End of dataset reached")


class OCRTools(QFrame):
    sign_on_select_model = Signal(str)

    def __init__(self, ocr_models: list[str]):
        super().__init__()
        self.setFixedHeight(60)
        self.setMinimumWidth(600)

        self.ocr_models = ocr_models

        self.layout = QHBoxLayout()
        self.model_selection = QComboBox()
        self.model_selection.setFixedWidth(180)

        for model in self.ocr_models:
            self.model_selection.addItem(model)

        # self.model_selection.activated.connect(self.on_select_ocr_model)
        self.model_selection.currentIndexChanged.connect(self.on_select_ocr_model)

        self.kernel_size_label = QLabel("Kernel Size")
        self.kernel_size_value = QLineEdit()
        self.kernel_size_value.setValidator(QIntValidator(3, 30))
        self.kernel_size_value.setText("20")

        self.kernel_iterations_label = QLabel("Kernel Iterations")
        self.kernel_iterations_value = QLineEdit()
        self.kernel_iterations_value.setValidator(QIntValidator(1, 10))
        self.kernel_iterations_value.setText("2")

        self.spacer = QLabel()

        self.layout.addWidget(self.model_selection)
        #self.layout.addWidget(self.kernel_size_label)
        #self.layout.addWidget(self.kernel_size_value)
        #self.layout.addWidget(self.kernel_iterations_label)
        #self.layout.addWidget(self.kernel_iterations_value)
        self.layout.addWidget(self.spacer)
        self.setLayout(self.layout)

        self.init()

        self.setStyleSheet(
            """
            color: #ffffff;
            background-color: #100F0F;
            border: 2px solid #100F0F; 
            border-radius: 8px;
        """
        )

    def init(self):
        print(f"Default PhotiPath: {Config.DEFAULT_PHOTI_LOCAL_PATH}")
        Config.init_models()

    def on_select_ocr_model(self, index: int):
        print(f"Selecting OCR Model : {index}")
        self.sign_on_select_model.emit(self.ocr_models[index])


class Footer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.label = QLabel()
        self.app_dir = QLabel()
        self.app_dir.setText(os.getcwd())
        self.label.setText(Config.get_default_model())

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.app_dir)

        self.setLayout(self.layout)


class PTGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setScene(self.scene)
        self.setMinimumHeight(200)

        self.zoom_in_factor = 1.25
        self.zoom_clamp = False
        self.zoom = 10
        self.zoom_step = 1
        self.zoom_range = [0, 10]

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        self.default_scrollbar_policy = Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        self.setHorizontalScrollBarPolicy(self.default_scrollbar_policy)
        self.setVerticalScrollBarPolicy(self.default_scrollbar_policy)

        self.verticalScrollBar().setSliderPosition(1)
        self.horizontalScrollBar().setSliderPosition(1)

        self.v_scrollbar = QScrollBar(self)
        self.h_scrollbar = QScrollBar(self)

        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)

        self.v_scrollbar.setStyleSheet(
            """
                QScrollBar:vertical {
                border: none;
                background: rgb(45, 45, 68);
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
                background: rgb(45, 45, 68);
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

    def wheelEvent(self, event):
        q_pointf = event.globalPosition()
        #print(f"Wheel Event position: {q_pointf}")
        # store mouse scene position
        event_pos = event.globalPosition()
        old_pos = self.mapToScene(int(event_pos.x()), int(event_pos.y()))
        # print(f"Wheel Event position: {old_pos}")
        # TODO: Use this position for zooming
        zoom_out_factor = 1 / self.zoom_in_factor
        zoom_factor = self.zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = self.zoom_in_factor
            self.zoom = self.zoom_step

        else:
            zoom_factor = zoom_out_factor
            self.zoom = self.zoom_step

        self.scale(zoom_factor, zoom_factor)


class PTGraphicsScene(QGraphicsScene):
    # see: https://forum.qt.io/topic/101616/pyside2-qtcore-signal-object-has-no-attribute-emit/4
    left_click_signal = Signal(QPoint)
    left_release_signal = Signal(QPoint)
    right_click_signal = Signal(QPoint)
    right_release_signal = Signal(QPoint)

    clear_selections = Signal()

    def __init__(self, scene, width: int = 1200, height: int = 600, parent=None):
        super().__init__(parent)
        self.scene = scene
        self._background_color = QColor("#393939")
        self.scene_width = width
        self.scene_height = height
        self.set_scene(self.scene_width, self.scene_height)
        self._line_color = QColor("#2f2f2f")

        self._pen_light = QPen(self._line_color)
        self._pen_light.setWidth(10)

        self._bg_image = QPixmap("Assets/Themes/Dark/background_grid.jpg")
        self.setBackgroundBrush(self._bg_image)

    def set_scene(self, width: int, height: int) -> None:
        self.setSceneRect(0, 0, width, height)

    def add_item(self, item: QGraphicsItem, z_order: int):
        item.setZValue(z_order)

        self.addItem(item)
        print(f"ItemAdded, Current SceneRect: {self.sceneRect()}")

    def remove_item(self, item: QGraphicsItem):
        self.removeItem(item)
        print(f"ItemRemoved, Current SceneRect: {self.sceneRect()}")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            print(f"LeftClick")
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
    def __init__(self, width: int = 1200, height: int = 400):
        super().__init__()

        self.default_width = width
        self.default_height = height
        self.setMinimumHeight(200)

        self.current_width = self.default_width
        self.current_height = self.default_height

        self.gr_scene = PTGraphicsScene(
            self, width=self.current_width, height=self.current_height
        )
        self.view = PTGraphicsView(self.gr_scene)
        self.view.setScene(self.gr_scene)

        self.layout = QVBoxLayout()
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

    def set_preview(self, data: BudaOCRData):
        print(f"Canvas -> set preview")
        _last_pos = self.gr_scene.get_current_item_pos()
        print(f"placing preview at pos: {_last_pos}")
        self.gr_scene.clear()
        preview_item = ImagePreview(data.image_path, data.line_data)
        preview_item.setPos(_last_pos)
        self.gr_scene.add_item(preview_item, 1)
        """
        q_image = QImage(data.image_path)
        q_pix = QPixmap.fromImage(q_image)
        gt_item = QGraphicsPixmapItem(q_pix)
        gt_item.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.gr_scene.add_item(gt_item, 0)
        """

class ImageList(QListWidget):
    sign_on_selected_item = Signal(UUID)
    """
    https://stackoverflow.com/questions/64576846/how-to-paint-an-outline-when-hovering-over-a-qlistwidget-item
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
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
                background: rgb(45, 45, 68);
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

        self.setAutoScrollMargin(20)
        # self.scroll_bar.setObjectName("PalmTreeBar")

        # setting vertical scroll bar to it
        self.setVerticalScrollBar(self.v_scrollbar)
        self.setHorizontalScrollBar(self.h_scrollbar)
        # self.itemEntered.connect(self.on_hover_item)
        self.setStyleSheet(
            """
            background-color: #100F0F;
            border: 2px solid #100F0F;
            border-radius: 8px;
           
            """
        )

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(
            item
        )  # returns an instance of CanvasHierarchyEntry

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
            print("Qframe->enter")
        elif event.type() == QEvent.Type.HoverLeave:
            print("QFrame->leave")

        return super().event(event)


class ImageThumb(QFrame):
    def __init__(self, image_path: str, max_height: int = 140, parent=None):
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

        self._pen_hover = QPen(QColor("#00ffdd"))
        self._pen_hover.setWidth(6)

        self._pen_select = QPen(QColor("#ffad00"))
        self._pen_select.setWidth(6)

        self.source_img = QImage(self.current_width, 140, QImage.Format.Format_ARGB32)
        self.source_img.fill(Qt.GlobalColor.blue)

        self.dest_img = QImage(self.current_width, 140, QImage.Format.Format_ARGB32)
        self.dest_img.fill(Qt.GlobalColor.transparent)

        self.clip_path = QPainterPath()
        print(f"Thumb rect: {self.source_img.rect()}")
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
        print(f"ImageThumb -> resize_thumb: {new_width}")
        self.current_width = new_width - 20
        self.source_img = QImage(
            self.current_width, self.max_height, QImage.Format.Format_ARGB32
        )
        self.source_img.fill(Qt.GlobalColor.blue)

        self.dest_img = QImage(
            self.current_width, self.max_height, QImage.Format.Format_ARGB32
        )
        self.dest_img.fill(Qt.GlobalColor.transparent)

        self.clip_path = QPainterPath()
        print(f"Thumb rect: {self.source_img.rect()}")
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
    def __init__(self, guid: UUID, image_path: str, image_name: str = "image"):
        super().__init__()
        self.guid = guid
        self.image_path = image_path
        self.file_name = get_filename(image_path)
        self.thumb = ImageThumb(image_path)
        self.label = QLabel()
        self.label.setText(self.file_name)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.thumb)
        self.is_active = False
        # self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            print(f"QListWidget -> Resize: {event.size()}")
            self.thumb.resize_thumb(event.size().width())

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
        self.thumb.is_selected = True
        self.thumb.update()


class ImageGallery(QFrame):
    def __init__(self, viewmodel: BudaViewModel, controller: ImageController):
        super().__init__()
        self.view_model = viewmodel
        self.controller = controller

        self.setObjectName("ImageGallery")
        self.setMinimumHeight(600)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)
        # self.setContentsMargins(20, 20, 20, 20)
        self.setContentsMargins(10, 10, 10, 10)
        # build layout
        self.image_label = QLabel(self)
        self.image_pixmap = QPixmap("Assets/Themes/Dark/BDRC_Logo.png").scaled(
            QSize(140, 90), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(self.image_pixmap)

        self.layout = QVBoxLayout()
        self.spacer = QSpacerItem(280, 20)
        self.image_list = ImageList(self)
        self.layout.addWidget(self.image_label)
        self.layout.addItem(self.spacer)
        self.layout.addWidget(self.image_list)
        self.setLayout(self.layout)

        self.setStyleSheet(
            """
                    background-color: #1d1c1c;
                """
        )

        self.current_width = 280

        # connect signals
        self.view_model.dataChanged.connect(self.add_data)
        self.view_model.dataCleared.connect(self.clear_data)
        self.image_list.sign_on_selected_item.connect(self.handle_item_selection)

    def resizeEvent(self, event):
        if isinstance(event, QResizeEvent):
            _new_size = event.size()
            self.current_width = _new_size.width()

    def handle_item_selection(self, guid: UUID):
        print(f"ImageGallery->Handling Item selection: {guid}is")
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
                    item_widget.is_active = False

    def add_data(self, data: list[BudaOCRData]):
        for _data in data:
            image_item = QListWidgetItem()
            image_item.setSizeHint(QSize(280, 160))
            image_widget = ImageListWidget(
                _data.guid,
                _data.image_path,
                _data.image_name,
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

        self.setStyleSheet(
            """
                QScrollBar:vertical {
                border: none;
                background: rgb(45, 45, 68);
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

    def on_item_clicked(self, item: QListWidgetItem):
        _list_item_widget = self.itemWidget(
            item
        )  # returns an instance of CanvasHierarchyEntry

        if isinstance(_list_item_widget, TextListWidget):
            print("TextWidgetList -> selected Text Widget")
            # self.sign_on_selected_item.emit(_list_item_widget.guid)

    def mouseMoveEvent(self, e):
        item = self.itemAt(e.pos())
        if item is not None:
            print("TextWidgetList->ITEM")
            _list_item_widget = self.itemWidget(item)
            print(f"Item Hovered over: {_list_item_widget}")

        """if isinstance(_list_item_widget, ImageListWidget):
            _list_item_widget.is_hovered = True
        """

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            print("Qframe->enter")
        elif event.type() == QEvent.Type.Leave:
            print("QFrame->leave")

        return super().event(event)


class TextListWidget(QWidget):
    """
    Custom widget holding the actual text data
    """

    def __init__(self, text: str, font: str, font_size: int = 20):
        super().__init__()
        self.text = text
        self.font = font
        self.font_size = font_size
        self.label = QLabel()
        self.label.setFont(QFont(self.font, font_size))
        self.label.setText(self.text)
        self.is_hovered = False
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

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

        #self.setFixedHeight(300)
        self.setMinimumHeight(100)
        self.setMinimumWidth(600)

        self.text_widget_list = TextWidgetList()

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.text_widget_list)
        self.setLayout(self.layout)
        self.setStyleSheet(
            """
                    color: #ffffff;
                    background-color: #100F0F;
                    border: 2px solid #100F0F; 
                    border-radius: 8px;
    
                """
        )

    def update_text(self, text: list[str]):
        self.text_widget_list.clear()

        for text_line in text:
            text_line = self.converter.toUnicode(text_line)
            list_item = QListWidgetItem()
            list_item.setSizeHint(QSize(800, 80))
            text_widget = TextListWidget(text_line, self.current_font, self.font_size)

            self.text_widget_list.addItem(list_item)
            self.text_widget_list.setItemWidget(list_item, text_widget)

        for idx in range(self.text_widget_list.count()):
            if idx % 2 == 0:
                _qBrush = QBrush(QColor("#100f0f"))
            else:
                _qBrush = QBrush(QColor("#1d1c1c"))
            self.text_widget_list.item(idx).setBackground(_qBrush)

    def update_font(self, font_path: str):
        self.current_font = font_path

    def update_font_size(self, font_size: int):
        self.font_size = font_size
