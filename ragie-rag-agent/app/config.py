"""
Configuration module for the Ragie RAG Agent.
Loads environment variables and provides configuration settings.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ragie API settings
RAGIE_API_TOKEN = os.getenv("RAGIE_API_TOKEN")
RAGIE_SERVER_URL = os.getenv("RAGIE_SERVER_URL", "https://api.ragie.ai")

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# FastAPI settings
API_ENV = os.getenv("API_ENV", "development")
API_DEBUG = os.getenv("API_ENV", "development").lower() == "development"
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-key-change-in-production")
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Webhook settings
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL")

def validate_config() -> tuple[bool, Optional[str]]:
    """
    Validate the configuration settings.
    
    Returns:
        bool: True if configuration is valid, False otherwise
        str: Error message if configuration is invalid
    """
    if not RAGIE_API_TOKEN:
        return False, "RAGIE_API_TOKEN is not set in environment variables"
    
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is not set in environment variables"
    
    if API_ENV == "production" and API_SECRET_KEY == "dev-key-change-in-production":
        return False, "API_SECRET_KEY should be changed in production environment"
    
    return True, None 