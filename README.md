# ✍️ Signature Cleaner

A full-stack, production-ready toolkit for cleaning, extracting, and enhancing digital signatures from images.
Features a beautiful web app, REST API, and batch CLI tool. Perfect for individuals, businesses, and developers.

---

## 🚀 Features

* **Smart Processing:** Removes backgrounds, enhances signature clarity
* **Modern Web UI:** Responsive, drag & drop, live previews, history gallery, beautiful gradients
* **Presets:** Light, Medium, Dark, Pencil, and Scan modes
* **Batch CLI:** Process multiple files or folders from the command line
* **RESTful API:** Easily integrate into other apps or automate
* **Multiple Formats:** Export cleaned signatures as PNG or JPG
* **History:** View/download previous processed signatures
* **Production Ready:** FastAPI backend, Docker, configuration files
* **Cross-platform:** Works on Windows, macOS, Linux


---

## 🏁 Quick Start

### 1. Clone and Set Up

```bash
git clone https://github.com/shamspias/signature-cleaner.git
cd signature-cleaner
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Start the Server

```bash
python main.py
```

Open your browser to [http://localhost:8000](http://localhost:8000)

---

### 3. Use the Web App

* Drag & drop a signature image
* Adjust cleaning controls or pick a preset
* Preview and download the cleaned signature instantly
* View and manage your processing history

---

### 4. Batch Processing via CLI

Process one or many images from the command line:

```bash
python cli.py signature.jpg --preset light
python cli.py -d ./signatures/ -o ./cleaned/
```

See all options with `python cli.py --help`

---

### 5. Use as an API

Send an image to the API and get back a cleaned version:

```python
import requests
with open('signature.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/process',
        files={'file': f},
        data={'preset': 'medium'}
    )
    print(response.json()['processed_url'])
```

---

### 6. Run with Docker

```bash
docker-compose up --build
# Then visit http://localhost:8000
```

---

## ⚙️ Configuration

Edit `config.py` to change:

* Default presets
* Storage folders
* Maximum upload size
* App settings

---

## 🛠️ Dependencies

* FastAPI
* Uvicorn
* OpenCV (`opencv-python`)
* Pillow
* numpy
* python-multipart
* (See `requirements.txt` for details)

---

## ✨ Screenshots


---

## 📜 License

MIT License.
Open for contributions and feedback!

---

## 👤 Author

Developed by Shamsuddin Ahmed.
Contact or open an issue for questions, requests, or improvements.

---

**Enjoy fast, beautiful, and effective digital signature cleaning!**
