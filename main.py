from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from pathlib import Path
import base64

from config import Config

Config.init_directories()

app = FastAPI(title=Config.API_TITLE)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
MEDIA_DIR = Config.MEDIA_DIR
UPLOAD_DIR = Config.UPLOAD_DIR
PROCESSED_DIR = Config.PROCESSED_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media")


class SignatureProcessor:
    """Core signature processing logic"""

    @staticmethod
    def process_signature(
            image_data: np.ndarray,
            threshold: int = 180,
            smoothing: float = 1.0,
            padding: int = 20,
            invert: bool = False,
            noise_reduction: bool = True
    ) -> tuple[np.ndarray, dict]:
        """
        Process signature image to clean background and extract signature.
        Returns processed image and metadata.
        """
        # Convert to grayscale if needed
        if len(image_data.shape) == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_data.copy()

        # Store original dimensions
        original_height, original_width = gray.shape

        # Apply smoothing
        if smoothing > 0:
            kernel_size = int(smoothing * 2 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Apply threshold
        if invert:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Noise reduction
        if noise_reduction:
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Remove isolated pixels
            binary = SignatureProcessor._remove_isolated_pixels(binary)

        # Find contours and bounding box
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No signature found, return white image
            result = np.ones_like(binary) * 255
            metadata = {
                "signature_found": False,
                "original_size": (original_width, original_height),
                "processed_size": (original_width, original_height)
            }
            return result, metadata

        # Get bounding box
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        for contour in contours:
            # Filter out very small contours
            if cv2.contourArea(contour) < 10:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Apply padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(binary.shape[1], x_max + padding)
        y_max = min(binary.shape[0], y_max + padding)

        # Crop
        cropped = binary[y_min:y_max, x_min:x_max]

        # Invert to get black on white
        result = cv2.bitwise_not(cropped)

        metadata = {
            "signature_found": True,
            "original_size": (original_width, original_height),
            "processed_size": (result.shape[1], result.shape[0]),
            "bounding_box": {
                "x": int(x_min),
                "y": int(y_min),
                "width": int(x_max - x_min),
                "height": int(y_max - y_min)
            }
        }

        return result, metadata

    @staticmethod
    def _remove_isolated_pixels(binary: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
        """Remove isolated pixels that have fewer than min_neighbors black pixels around them."""
        result = binary.copy()
        height, width = binary.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if binary[y, x] == 0:  # Black pixel
                    # Count black neighbors
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if binary[y + dy, x + dx] == 0:
                                neighbors += 1

                    if neighbors < min_neighbors:
                        result[y, x] = 255  # Make it white

        return result


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")


@app.post("/api/process")
async def process_signature(
        file: UploadFile = File(...),
        threshold: int = Form(180),
        smoothing: float = Form(1.0),
        padding: int = Form(20),
        invert: bool = Form(False),
        noise_reduction: bool = Form(True)
):
    """Process uploaded signature image"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        original_filename = f"original_{timestamp}_{unique_id}.png"
        processed_filename = f"processed_{timestamp}_{unique_id}.png"

        # Save original
        original_path = UPLOAD_DIR / original_filename
        cv2.imwrite(str(original_path), img)

        # Process signature
        processed_img, metadata = SignatureProcessor.process_signature(
            img,
            threshold=threshold,
            smoothing=smoothing,
            padding=padding,
            invert=invert,
            noise_reduction=noise_reduction
        )

        # Save processed
        processed_path = PROCESSED_DIR / processed_filename
        cv2.imwrite(str(processed_path), processed_img)

        # Convert processed image to base64 for preview
        _, buffer = cv2.imencode('.png', processed_img)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "success": True,
            "original_url": f"/media/uploads/{original_filename}",
            "processed_url": f"/media/processed/{processed_filename}",
            "processed_preview": f"data:image/png;base64,{processed_base64}",
            "metadata": metadata,
            "parameters": {
                "threshold": threshold,
                "smoothing": smoothing,
                "padding": padding,
                "invert": invert,
                "noise_reduction": noise_reduction
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/presets")
async def get_presets():
    """Get available presets"""
    return {
        "presets": {
            "light": {
                "name": "Light Signature",
                "threshold": 220,
                "smoothing": 0.5,
                "invert": False
            },
            "medium": {
                "name": "Medium Signature",
                "threshold": 180,
                "smoothing": 1.0,
                "invert": False
            },
            "dark": {
                "name": "Dark Signature",
                "threshold": 140,
                "smoothing": 1.5,
                "invert": False
            },
            "pencil": {
                "name": "Pencil Signature",
                "threshold": 200,
                "smoothing": 0,
                "invert": False
            },
            "scan": {
                "name": "Scanned Document",
                "threshold": 160,
                "smoothing": 2.0,
                "invert": False
            }
        }
    }


@app.get("/api/history")
async def get_history(limit: int = 10):
    """Get recent processed images"""
    processed_files = []

    for file in sorted(PROCESSED_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)[:limit]:
        # Try to find corresponding original
        original_pattern = file.name.replace("processed_", "original_")
        original_file = list(UPLOAD_DIR.glob(original_pattern))

        processed_files.append({
            "processed_url": f"/media/processed/{file.name}",
            "original_url": f"/media/uploads/{original_file[0].name}" if original_file else None,
            "timestamp": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
            "filename": file.name
        })

    return {"files": processed_files}


@app.delete("/api/clear-history")
async def clear_history():
    """Clear all processed images"""
    try:
        # Clear uploads
        for file in UPLOAD_DIR.glob("*"):
            if file.is_file():
                file.unlink()

        # Clear processed
        for file in PROCESSED_DIR.glob("*"):
            if file.is_file():
                file.unlink()

        return {"success": True, "message": "History cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Signature Cleaner API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.HOST, port=Config.PORT, reload=Config.RELOAD)
