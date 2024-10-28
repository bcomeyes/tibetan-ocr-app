from dataclasses import dataclass

@dataclass
class BaseTheme:
    bdrc_logo = "Assets/Textures/BDRC_Logo.png"
    import_btn = "Assets/Textures/import.png"
    new_btn = "Assets/Textures/new_light.png"
    play_btn = "Assets/Textures/play_light.png"
    save_btn = "Assets/Textures/save-disc.png"
    next_btn = "Assets/Textures/next.png"
    prev_btn = "Assets/Textures/prev.png"
    settings_btn = "Assets/Textures/settings.png"

@dataclass
class Light(BaseTheme):
    hover_color: str

@dataclass
class Dark(BaseTheme):
    background: str
    header_btn_hover: str


DARK = """
    QWidget#MainWindow {
        background-color: #1d1c1c;
        color: #000000;
    }

    QFrame#HeaderTools {
        background-color: #100F0F;
        min-width: 880px;
        margin-left: 0px;
        alignment: left;
        border: 2px solid #100F0F; 
        border-radius: 6px;
    }

    QWidget#ToolBox {
        color: #ffffff;
        background-color: #100F0F;
        alignment-left;
    }

    QWidget#PageSwitcher {
        color: #ffffff;
        background-color: #100F0F;
        border: 2px solid #100F0F; 
        border-radius: 8px;
    }

    QWidget#PageSelector {
        border: 1px solid white;
        border-radius: 4px;
    }

    QComboBox#ModelSelection {
        background: #434343;
        min-width: 220px;
        border: 2px solid #ced4da;
        border-radius: 4px;
    }

    QPushButton#MenuButton {
        background-color: #172832;
        border: 1px solid #1d1d1d;
        border-radius: 4px;
    }

    QPushButton#MenuButton::QIcon {
        color: #d73449;
    }

    QPushButton#DialogButton {
        background-color: #A40021;
        border-radius: 4px;
        height: 24;
    }

    QPushButton#DialogButton:hover {
        color: #ffad00;
    }
    
    QPushButton#SmallDialogButton {
        background-color: #A40021;
        border-radius: 4px;
        height: 20;
        width: 80px;
    }
    
    QPushButton#SmallDialogButton:hover {
        color: #ffad00;
    }
    """