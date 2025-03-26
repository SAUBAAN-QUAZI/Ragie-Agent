"""
Main FastAPI application for the Ragie RAG Agent.
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io

from app.config import ALLOWED_ORIGINS, validate_config, API_DEBUG, WEBHOOK_BASE_URL
from app.ragie_client import upload_document, retrieve_chunks, generate_response, setup_webhook, list_documents, get_ragie_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set debug level for development
if API_DEBUG:
    logger.setLevel(logging.DEBUG)
    logging.getLogger("app").setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# Create FastAPI app
app = FastAPI(
    title="Ragie RAG Agent",
    description="A Retrieval-Augmented Generation system built with Ragie SDK",
    version="0.1.0",
    debug=API_DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to validate configuration
async def verify_config():
    is_valid, error_message = validate_config()
    if not is_valid:
        logger.error(f"Configuration error: {error_message}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: {error_message}")
    return True

# Models for request/response objects
class QueryRequest(BaseModel):
    query: str
    client_id: str
    top_k: int = 8

class DocumentListRequest(BaseModel):
    client_id: str

class WebhookEvent(BaseModel):
    event_type: str
    document_id: Optional[str] = None
    status: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

# Setup webhook on startup if configured
@app.on_event("startup")
async def startup_event():
    logger.info("Ragie RAG Agent starting up...")
    
    # Validate configuration
    is_valid, error_message = validate_config()
    if not is_valid:
        logger.error(f"Configuration error: {error_message}")
        # Continue startup, but API calls may fail
    
    if WEBHOOK_BASE_URL:
        try:
            logger.info(f"Setting up webhook at {WEBHOOK_BASE_URL}/webhook")
            result = setup_webhook(f"{WEBHOOK_BASE_URL}/webhook")
            logger.info(f"Webhook setup result: {result}")
            if result.get("status") == "warning":
                logger.warning(result.get("message", "Webhook setup warning"))
        except Exception as e:
            logger.error(f"Failed to set up webhook: {str(e)}")
            logger.info("Continuing startup without webhook setup")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running and Ragie client is available."""
    health_status = {
        "status": "healthy",
        "api": "online",
        "ragie_client": "unknown"
    }
    
    # Check Ragie client connection
    try:
        client = get_ragie_client()
        # Just initialize the client to check if it works
        with client as _:
            health_status["ragie_client"] = "connected"
    except Exception as e:
        logger.warning(f"Ragie client health check failed: {str(e)}")
        health_status["ragie_client"] = "error"
        health_status["ragie_error"] = str(e)
    
    return health_status

# Upload document endpoint
@app.post("/upload", dependencies=[Depends(verify_config)])
async def upload_file(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    try:
        logger.info(f"Uploading file {file.filename} for client {client_id}")
        
        # Validate file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["pdf", "docx", "csv", "txt"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported types: pdf, docx, csv, txt"
            )
        
        # Read file content
        contents = await file.read()
        logger.debug(f"Read {len(contents)} bytes from file {file.filename}")
        
        # Create a BytesIO object from contents
        file_obj = io.BytesIO(contents)
        file_obj.name = file.filename
        
        # Try setting the file pointer to the beginning
        file_obj.seek(0)
        
        # Upload to Ragie
        try:
            response = upload_document(
                file=file_obj,
                client_id=client_id,
                file_name=file.filename
            )
            
            logger.info(f"Upload response: {response}")
            return response
        except Exception as e:
            # If there's an issue with the file object, try another approach
            logger.warning(f"Error with file object, trying with raw bytes: {str(e)}")
            
            # Try directly with raw bytes
            file_obj = contents
            response = upload_document(
                file=file_obj,
                client_id=client_id,
                file_name=file.filename
            )
            
            logger.info(f"Upload with raw bytes response: {response}")
            return response
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/query", dependencies=[Depends(verify_config)])
async def query(request: QueryRequest):
    try:
        logger.info(f"Processing query for client {request.client_id}: {request.query}")
        
        # Retrieve chunks from Ragie
        retrieval_result = retrieve_chunks(
            query=request.query,
            client_id=request.client_id,
            top_k=request.top_k
        )
        
        # Extract chunks from response - handle both "chunks" and "scored_chunks" fields
        chunks = retrieval_result.get("chunks", retrieval_result.get("scored_chunks", []))
        
        if not chunks:
            logger.warning(f"No chunks found for query: {request.query}")
            return {
                "answer": "No relevant information found for your query.",
                "citations": []
            }
        
        # Generate response with citations
        response = generate_response(
            query=request.query,
            chunks=chunks
        )
        
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# List documents endpoint
@app.post("/documents")
async def get_documents(request: DocumentListRequest):
    try:
        logger.info(f"Listing documents for client {request.client_id}")
        documents = list_documents(client_id=request.client_id)
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Webhook handler endpoint
@app.post("/webhook")
async def webhook_handler(event: WebhookEvent):
    try:
        logger.info(f"Received webhook event: {event.event_type}")
        
        if event.event_type == "document_status_updated":
            logger.info(f"Document {event.document_id} status updated to {event.status}")
            # Here you could update a database, notify users, etc.
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        ) 