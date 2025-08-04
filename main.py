from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
import base64
from typing import Optional

from config import Config

# Import the enhanced processor - you'll need to save the enhanced_processor code as a separate file
try:
    from enhanced_processor import EnhancedSignatureProcessor
except ImportError:
    # Fallback to inline definition if file not created yet
    exec(open('enhanced_processor.py').read()) if os.path.exists('enhanced_processor.py') else None

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

# Initialize enhanced processor
processor = EnhancedSignatureProcessor()


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image and suggest processing parameters"""
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

        # Analyze image
        analysis = processor.analyze_image(img)

        return JSONResponse({
            "success": True,
            "analysis": analysis
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process")
async def process_signature(
        file: UploadFile = File(...),
        method: str = Form("adaptive"),
        threshold: int = Form(180),
        smoothing: float = Form(1.0),
        padding: int = Form(20),
        invert: bool = Form(False),
        noise_reduction: bool = Form(True),
        enhance_contrast: bool = Form(True),
        min_signature_area: int = Form(100)
):
    """Process uploaded signature image with enhanced options"""
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

        # Process signature with enhanced processor
        processed_img, metadata = processor.process_signature(
            img,
            method=method,
            threshold=threshold,
            smoothing=smoothing,
            padding=padding,
            invert=invert,
            noise_reduction=noise_reduction,
            enhance_contrast=enhance_contrast,
            min_signature_area=min_signature_area
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
                "method": method,
                "threshold": threshold,
                "smoothing": smoothing,
                "padding": padding,
                "invert": invert,
                "noise_reduction": noise_reduction,
                "enhance_contrast": enhance_contrast,
                "min_signature_area": min_signature_area
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-batch")
async def process_batch(
        files: list[UploadFile] = File(...),
        method: str = Form("adaptive"),
        threshold: int = Form(180),
        smoothing: float = Form(1.0),
        padding: int = Form(20),
        invert: bool = Form(False),
        noise_reduction: bool = Form(True),
        enhance_contrast: bool = Form(True)
):
    """Process multiple signatures at once"""
    results = []

    for file in files[:10]:  # Limit to 10 files
        try:
            # Read and decode image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image file"
                })
                continue

            # Process signature
            processed_img, metadata = processor.process_signature(
                img,
                method=method,
                threshold=threshold,
                smoothing=smoothing,
                padding=padding,
                invert=invert,
                noise_reduction=noise_reduction,
                enhance_contrast=enhance_contrast
            )

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            processed_filename = f"batch_{timestamp}_{unique_id}.png"

            # Save processed
            processed_path = PROCESSED_DIR / processed_filename
            cv2.imwrite(str(processed_path), processed_img)

            results.append({
                "filename": file.filename,
                "success": True,
                "processed_url": f"/media/processed/{processed_filename}",
                "metadata": metadata
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return JSONResponse({"results": results})


@app.get("/api/presets")
async def get_presets():
    """Get available presets with enhanced options"""
    return {
        "presets": {
            "light": {
                "name": "Light Signature",
                "method": "adaptive",
                "threshold": 220,
                "smoothing": 0.5,
                "invert": False,
                "enhance_contrast": True
            },
            "medium": {
                "name": "Medium Signature",
                "method": "adaptive",
                "threshold": 180,
                "smoothing": 1.0,
                "invert": False,
                "enhance_contrast": True
            },
            "dark": {
                "name": "Dark Signature",
                "method": "otsu",
                "threshold": 140,
                "smoothing": 1.5,
                "invert": False,
                "enhance_contrast": True
            },
            "pencil": {
                "name": "Pencil Signature",
                "method": "adaptive",
                "threshold": 200,
                "smoothing": 0,
                "invert": False,
                "enhance_contrast": True
            },
            "scan": {
                "name": "Scanned Document",
                "method": "otsu",
                "threshold": 160,
                "smoothing": 2.0,
                "invert": False,
                "enhance_contrast": False
            },
            "photo": {
                "name": "Photo Signature",
                "method": "ml",
                "threshold": 180,
                "smoothing": 1.0,
                "invert": False,
                "enhance_contrast": True
            }
        },
        "methods": {
            "manual": "Manual threshold (classic method)",
            "adaptive": "Adaptive threshold (best for varying lighting)",
            "otsu": "Otsu's method (automatic threshold)",
            "ml": "ML-enhanced (experimental)"
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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Signature Cleaner API", "version": "2.0"}


# Backward compatibility endpoint
@app.post("/api/process-simple")
async def process_signature_simple(
        file: UploadFile = File(...),
        threshold: int = Form(180),
        smoothing: float = Form(1.0),
        padding: int = Form(20),
        invert: bool = Form(False),
        noise_reduction: bool = Form(True)
):
    """Original simple processing endpoint for backward compatibility"""
    return await process_signature(
        file=file,
        method="manual",
        threshold=threshold,
        smoothing=smoothing,
        padding=padding,
        invert=invert,
        noise_reduction=noise_reduction,
        enhance_contrast=False,
        min_signature_area=100
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.HOST, port=Config.PORT, reload=Config.RELOAD)
