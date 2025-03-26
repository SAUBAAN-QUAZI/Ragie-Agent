"""
Run script for the Ragie RAG Agent FastAPI application.
"""
import uvicorn
from app.config import API_PORT, API_DEBUG

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=API_DEBUG
    ) 