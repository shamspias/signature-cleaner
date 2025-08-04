import os
from pathlib import Path


class Config:
    """Application configuration"""

    # Base paths
    BASE_DIR = Path(__file__).parent
    MEDIA_DIR = BASE_DIR / "media"
    UPLOAD_DIR = MEDIA_DIR / "uploads"
    PROCESSED_DIR = MEDIA_DIR / "processed"
    STATIC_DIR = BASE_DIR / "static"

    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    RELOAD = os.getenv("RELOAD", "True").lower() == "true"

    # Upload settings
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10MB default
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}

    # Processing defaults
    DEFAULT_THRESHOLD = int(os.getenv("DEFAULT_THRESHOLD", 180))
    DEFAULT_SMOOTHING = float(os.getenv("DEFAULT_SMOOTHING", 1.0))
    DEFAULT_PADDING = int(os.getenv("DEFAULT_PADDING", 20))

    # History settings
    MAX_HISTORY_ITEMS = int(os.getenv("MAX_HISTORY_ITEMS", 50))
    AUTO_CLEANUP_DAYS = int(os.getenv("AUTO_CLEANUP_DAYS", 7))  # Auto-delete after 7 days

    # Security
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

    # Others
    API_TITLE = os.getenv("API_TITLE", "Signature Cleaner API")

    @classmethod
    def init_directories(cls):
        """Create necessary directories"""
        cls.MEDIA_DIR.mkdir(exist_ok=True)
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.PROCESSED_DIR.mkdir(exist_ok=True)
        cls.STATIC_DIR.mkdir(exist_ok=True)
