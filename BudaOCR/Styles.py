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
        
        QLineEdit {
            background: #434343;
        }
    }

    QWidget#PageSelector {
        border: 1px solid white;
        border-radius: 4px;
    }

    QComboBox#ModelSelection {
        background: #434343;
        color: #ffffff;
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

    
    QPushButton#TextToolsButton {
        background-color: #3f3f3f;
        border: 2px solid #1d1d1d;
        border-radius: 4px;
    }
    
    QPushButton#TextToolsButton:hover {
        color: #ffad00;
    }
    

    QDialog#ExportDialog {
        color: #ffffff;
        background-color: #1d1c1c;
    }
    
    QDialog#SettingsDialog {
        color: #ffffff;
        background-color: #100F0F;
        
        QTabWidget::tab-bar {
                left: 20px;
                height: 36px;
                border-radius: 4px;
                alignment: center; 
        }      
    }
    
    QDialog#BatchOCRDialog
    {
        color: #ffffff;
        background-color: #100F0F;
    }
    
    
    QPushButton#DialogButton {
        color: #ffffff;
        background-color: #A40021;
        border-radius: 4px;
        height: 24;
    }

    QPushButton#DialogButton:hover {
        color: #ffad00;
    }
    
    QPushButton#SmallDialogButton {
        color: #ffffff;
        background-color: #A40021;
        border-radius: 4px;
        height: 20;
        width: 80px;
    }
    
    QPushButton#SmallDialogButton:hover {
        color: #ffad00;
    }
    
    QRadioButton#OptionsRadio
    {
        color: #ffffff;
        border: none;
    }
    
    QButtonGroup#OptionsRadio
    {
        color: #ffffff;
        border: none;
    }
    
    QLabel#OptionsLabel
    {
        color: #ffffff;
    }
    
    QLineEdit#DialogLineEdit {
        color: #ffffff;
        background-color: #474747;
        border: 2px solid #343942;
        border-radius: 2px;
        text-align: left;
    }
    
    QProgressBar#DialogProgressBar {
        background-color: #474747;
        border: 2px solid #343942;
        border-radius: 2px;
    
        QProgressBar::chunk {
            background-color: #2c2f37;
            width: 20px;
}
    }
    
    """