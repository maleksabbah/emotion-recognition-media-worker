"""Root entry point — runs the media worker."""
import asyncio
from app.main import run

if __name__ == "__main__":
    asyncio.run(run())