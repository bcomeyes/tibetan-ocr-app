# Tibetan OCR local app

This app is a free and open source offline App that can be installed on a local compter to run Tibetan OCR on batches of images (including in PDFs). It was developed by Eric Werner for the [Buddhist Digital Resource Center](https://www.bdrc.io).

### Main features

The app can open one or multiple files and run OCR on them. It can export plain text or [PageXML](https://github.com/PRImA-Research-Lab/PAGE-XML) (a format it shares with [Transkribus](https://www.transkribus.org/)). 

It can also optionally dewarp images as well as convert the output to Wylie.

Instead of providing one model that can handle all styles of Tibetan writing, we provide a few different models that we encourage users to experiment with to see what fits their data best.

### Installation and running

##### Windows

1. Download and unzip https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.1/bdrc_ocr_win64_0.1.zip
2. Download and unzip https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.1/bdrc-ocr-app-models.zip in the `Resources/` folder of the directory you unzipped in step 1
3. Run `bdrc.exe`

##### OSX

1. Download and unzip https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.1/bdrc_ocr_0.1_macos_arm64.zip
2. Download and unzip https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.1/bdrc-ocr-app-models.zip in a folder of your choice
3. Run the `BDRC.app` folder

##### From source / Linux

1. Clone the Github repository: `git clone https://github.com/buda-base/tibetan-ocr-app.git`
2. Download and unzip https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.1/bdrc-ocr-app-models.zip in the `Resources/` folder
3. Install dependencies with `pip install -r requirements.txt` (requires at least Python 3.10)
4. Run `python main.py`

### Configuring the models

Once the app opens, click on the setting icon and select:
- the `Resource/OCRModels/` folder on Windows and from source / Linux
- the `OCRModels/` directory of the folder where you unzipped the model files on OSX
