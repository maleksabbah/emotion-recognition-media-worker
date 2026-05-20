"""Root entry point — re-exports the FastAPI app from app/main.py."""
from app.main import app

__all__ = ["app"]