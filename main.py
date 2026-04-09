"""Media Worker entry point for Docker."""
import asyncio
from app.Worker import main

if __name__ == "__main__":
    asyncio.run(main())