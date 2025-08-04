# ✍️ Signature Cleaner

A professional FastAPI web application to **clean digital signatures**, remove backgrounds, and extract the signature on a pure white background—complete with smart auto-cropping, batch tools, API access, and a beautiful frontend.

---

## 🚀 Features

* 🎨 **Smart Background Removal** – Detects and removes paper noise or shadows automatically
* ✂️ **Auto-Cropping** – Extracts just the signature with customizable padding
* 🔧 **Advanced Controls** – Fine-tune threshold, smoothing, padding, invert, and noise reduction
* 🎯 **Presets** – One-click settings for light, dark, pencil, or scanned signatures
* 📱 **Responsive Design** – Works beautifully on desktop, tablet, and mobile
* 💾 **Multiple Export Formats** – Download cleaned signatures as PNG or JPG
* 📚 **History Gallery** – Manage and view previously processed signatures
* ⚡ **Fast Processing** – Powered by OpenCV for instant results
* 🛠️ **Batch CLI & REST API** – Integrate or automate with CLI or direct API calls
* 🐳 **Docker-Ready** – Easy deployment anywhere

---

## ⚡ Quickstart

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

### 2. Start the Server

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open in Browser

Go to: [http://localhost:8000](http://localhost:8000)
Enjoy the clean, responsive signature cleaning interface!

---

## 🖥️ Web Usage

1. **Upload**: Drag & drop or click to upload a signature image (JPG, PNG, WEBP).
2. **Adjust**: Use controls to tweak threshold, smoothing, padding, etc., or pick a preset.
3. **Process**: Click **Process Signature**.
4. **Download**: Save your cleaned signature as PNG or JPG.
5. **History**: Access and manage previously processed signatures in the gallery.

---

## 🔌 API Usage

**Process a signature via API:**

```python
import requests

with open('signature.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'threshold': 180,
        'smoothing': 1.0,
        'padding': 20,
        'invert': False,
        'noise_reduction': True
    }
    response = requests.post('http://localhost:8000/api/process', files=files, data=data)
    print(response.json()['processed_url'])
```

### API Endpoints

| Method | Path                 | Description               |
| ------ | -------------------- | ------------------------- |
| GET    | `/`                  | Web UI                    |
| POST   | `/api/process`       | Process uploaded image    |
| GET    | `/api/presets`       | List of available presets |
| GET    | `/api/history`       | Processed image gallery   |
| DELETE | `/api/clear-history` | Clear gallery/history     |
| GET    | `/health`            | Health check              |

---

## ⚙️ Configuration

* Edit variables in `main.py` or use `config.py` (if present) to customize presets, folder locations, upload size limits, etc.
* The app will auto-create `media/uploads/` and `media/processed/` folders.

---

## 🛠️ Batch Processing (CLI)

Want to process multiple signatures at once?

```bash
python cli.py signature.jpg --preset light
python cli.py -d ./signatures/ -o ./cleaned/
```

See all CLI options with: `python cli.py --help`

---

## 🐳 Docker Deployment (Optional)

### 1. **Create a Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. **Build & Run**

```bash
docker build -t signature-cleaner .
docker run -p 8000:8000 -v $(pwd)/media:/app/media signature-cleaner
```

---

## ☁️ Production Notes

* Use Nginx as a reverse proxy for security & SSL.
* Configure `media/` storage according to your infra.
* Deploy with Gunicorn & Uvicorn for production:

  ```bash
  gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
  ```

---

## ❗ Troubleshooting

* **OpenCV errors:** If issues arise, try `pip install opencv-python-headless` or install OpenCV system-wide.
* **Permissions:** Make sure the `media/` directory is writable.
* **Large Files:** Increase limits in FastAPI or your proxy if needed.

---

## 📜 License

MIT License – Free for personal or commercial use.

---

## 👤 Author

Created by Shamsuddin Ahmed

---

**Happy signature cleaning! ✨**
