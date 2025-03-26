"""
Ragie SDK client module.
Provides a client for interacting with the Ragie API.
"""
from functools import lru_cache
from typing import Optional, Dict, Any, List
import json
import logging
from datetime import datetime
import requests

from ragie import Ragie, models
from openai import OpenAI

from app.config import RAGIE_API_TOKEN, RAGIE_SERVER_URL, OPENAI_API_KEY, OPENAI_MODEL

# Configure logger
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@lru_cache()
def get_ragie_client() -> Ragie:
    """
    Get a cached Ragie client instance.
    
    Returns:
        Ragie: A Ragie client instance
    """
    return Ragie(
        auth=RAGIE_API_TOKEN,
        server_url=RAGIE_SERVER_URL
    )

def upload_document(file, client_id: str, file_name: str) -> Dict[str, Any]:
    """Upload a document to Ragie using direct API call."""
    url = f"{RAGIE_SERVER_URL}/documents"
    partition = f"client_{client_id}"
    
    # Create form data
    files = {
        "file": (file_name, file, "application/octet-stream")
    }
    
    data = {
        "partition": partition,
        "metadata": json.dumps({
            "client_id": client_id,
            "upload_timestamp": datetime.now().isoformat(),
            "source": "rag_agent"
        })
    }
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {RAGIE_API_TOKEN}"
    }
    
    try:
        logger.debug(f"Uploading document to Ragie API: {file_name}")
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise

def retrieve_chunks(query: str, client_id: str, top_k: int = 8) -> Dict[str, Any]:
    """Retrieve relevant document chunks using direct API call."""
    url = f"{RAGIE_SERVER_URL}/retrievals"
    partition = f"client_{client_id}"
    
    payload = {
        "query": query,
        "partition": partition,
        "top_k": top_k,
        "rerank": True,
        "recency_bias": True
    }
    
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {RAGIE_API_TOKEN}"
    }
    
    try:
        logger.debug(f"Retrieving chunks for query: {query}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # Map scored_chunks to chunks for consistent naming
        if "scored_chunks" in result and "chunks" not in result:
            result["chunks"] = result["scored_chunks"]
            
        return result
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return {"chunks": []}

def generate_response(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a response using the retrieved chunks.
    
    Args:
        query: User query
        chunks: Retrieved document chunks
    
    Returns:
        Dict: Response with answer and citations
    """
    # Prepare context from chunks
    context_parts = []
    citations = []
    
    for i, chunk in enumerate(chunks):
        # Add chunk to context
        context_parts.append(f"[{i+1}] {chunk.get('text', '')}")
        
        # Prepare citation
        citation = {
            "id": i+1,
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", "")[:100] + "...",  # Truncate for display
            "page_number": chunk.get("page_number", "N/A"),
            "score": chunk.get("score", 0)
        }
        citations.append(citation)
    
    context = "\n\n".join(context_parts)
    
    # Generate response using OpenAI
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always include citations to your sources using the format [n] where n is the citation number.
If the answer cannot be found in the context, state that clearly."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer the question based on the context provided. Use citations in the format [n] to reference your sources."""
    
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "citations": citations
    }

def setup_webhook(endpoint_url: str) -> Dict[str, Any]:
    """
    Set up a webhook to receive document status updates.
    
    Note: According to documentation, webhook endpoints are managed in the Ragie UI.
    This function is kept for compatibility but will log a warning.
    
    Args:
        endpoint_url: URL to receive webhook events
    
    Returns:
        Dict: Message about webhook setup
    """
    logger.warning(
        "Webhook setup via API may not be supported. Please set up webhooks in the Ragie UI: "
        "https://app.ragie.ai - See documentation at https://docs.ragie.ai/docs/webhooks"
    )
    
    # Attempt to set up webhook if API is available
    try:
        with get_ragie_client() as client:
            # Check if the client has the webhooks attribute
            if hasattr(client, 'webhooks') and hasattr(client.webhooks, 'create'):
                webhook_config = {
                    "name": "RAG Agent Webhook",
                    "url": endpoint_url,
                    "active": True,
                    "events": ["document_status_updated", "document_deleted"]
                }
                
                response = client.webhooks.create(request=webhook_config)
                return response.to_dict()
    except Exception as e:
        logger.error(f"Failed to set up webhook programmatically: {str(e)}")
    
    # Return a message instead of raising an exception
    return {
        "status": "warning",
        "message": "Webhook setup via API may not be supported. Please set up webhooks in the Ragie UI."
    }

def list_documents(client_id: str) -> List[Dict[str, Any]]:
    """List documents for a client using direct API call."""
    url = f"{RAGIE_SERVER_URL}/documents"
    partition = f"client_{client_id}"
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {RAGIE_API_TOKEN}",
        "partition": partition
    }
    
    try:
        logger.debug(f"Listing documents for client: {client_id}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Log response for debugging
        logger.debug(f"API Response: {response.status_code}")
        
        # Parse JSON
        try:
            result = response.json()
            logger.debug(f"Response structure: {type(result)}")
            
            # Handle different response formats
            if isinstance(result, list):
                return result  # List of documents
            elif isinstance(result, dict):
                if "documents" in result:
                    return result["documents"]  # Dict with documents key
                else:
                    # Handle case where a single document is returned
                    return [result]
            else:
                logger.warning(f"Unexpected response type: {type(result)}")
                return []
                
        except json.JSONDecodeError:
            # Log the raw response if JSON parsing fails
            logger.error(f"Failed to parse JSON: {response.text[:200]}...")
            return []
            
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response code: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text[:200]}...")
        return []
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return [] 