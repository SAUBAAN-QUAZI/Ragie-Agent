"""
Streamlit frontend for the Ragie RAG Agent.
"""
import os
import json
import tempfile
import streamlit as st
import requests
from typing import List, Dict, Any
from datetime import datetime

# Configure the Streamlit page
st.set_page_config(
    page_title="Ragie RAG Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_CLIENTS = ["client1", "client2", "client3"]  # Can be replaced with dynamic client fetching

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_client" not in st.session_state:
    st.session_state.current_client = None

if "documents" not in st.session_state:
    st.session_state.documents = []

# Add polling for document status
if "last_poll_time" not in st.session_state:
    st.session_state.last_poll_time = datetime.now()

# Helper function to get progress percentage based on status
def get_progress_percentage(status):
    """Convert document status to progress percentage"""
    status_map = {
        "pending": 10,
        "partitioning": 30,
        "chunking": 50, 
        "chunked": 60,
        "indexing": 70,
        "indexed": 80,
        "keyword_indexed": 90,
        "ready": 100,
        "failed": 0,
        "unknown": 0
    }
    return status_map.get(status.lower(), 20)  # Default to 20% if status unknown

# Auto-refresh documents status every 15 seconds
def auto_refresh_documents():
    """Auto-refresh document status"""
    now = datetime.now()
    time_diff = (now - st.session_state.last_poll_time).total_seconds()
    
    # Only refresh every 15 seconds to avoid too many API calls
    if time_diff > 15 and st.session_state.current_client:
        st.session_state.documents = list_documents(st.session_state.current_client)
        st.session_state.last_poll_time = now
        return True
    return False

# Helper functions to interact with the FastAPI backend
def upload_file(file, client_id: str) -> Dict[str, Any]:
    """
    Upload a file to the RAG Agent API.
    
    Args:
        file: File object from Streamlit
        client_id: Client identifier
        
    Returns:
        Dict: API response
    """
    url = f"{API_BASE_URL}/upload"
    
    # Create form data
    files = {"file": file}
    data = {"client_id": client_id}
    
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading file: {str(e)}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Server response: {e.response.text}")
        return None

def query_documents(query: str, client_id: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Query documents via the RAG Agent API.
    
    Args:
        query: User query
        client_id: Client identifier
        top_k: Number of chunks to retrieve
        
    Returns:
        Dict: API response with answer and citations
    """
    url = f"{API_BASE_URL}/query"
    
    payload = {
        "query": query,
        "client_id": client_id,
        "top_k": top_k
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying documents: {str(e)}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Server response: {e.response.text}")
        return None

def list_documents(client_id: str) -> List[Dict[str, Any]]:
    """
    List documents for a client.
    
    Args:
        client_id: Client identifier
        
    Returns:
        List: List of documents
    """
    url = f"{API_BASE_URL}/documents"
    
    payload = {
        "client_id": client_id
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error listing documents: {str(e)}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Server response: {e.response.text}")
        return []

def health_check() -> bool:
    """
    Check if the API is healthy.
    
    Returns:
        bool: True if healthy, False otherwise
    """
    url = f"{API_BASE_URL}/health"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return True
    except:
        return False

# Function to display chat messages
def display_chat_message(message, is_user=False):
    """Display a chat message with the appropriate styling."""
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: #2e7eea; color: white; padding: 10px 15px; border-radius: 15px 15px 0 15px; max-width: 80%;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
            <div style="background-color: #f0f0f0; color: #111111; padding: 10px 15px; border-radius: 15px 15px 15px 0; max-width: 80%;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Function to display citations
def display_citations(citations):
    """Display citations in an expandable section."""
    with st.expander("📚 View Sources"):
        for i, citation in enumerate(citations):
            st.markdown(f"**Source {i+1}**: {citation.get('text', '')}")
            st.markdown(f"Document ID: `{citation.get('document_id', 'N/A')}`")
            if citation.get('page_number', 'N/A') != 'N/A':
                st.markdown(f"Page: {citation.get('page_number', 'N/A')}")
            st.markdown(f"Confidence Score: {citation.get('score', 0):.2f}")
            st.markdown("---")

# Sidebar for client selection and file upload
with st.sidebar:
    st.title("📚 Ragie RAG Agent")
    
    # API health check
    if health_check():
        st.success("✅ API is online")
    else:
        st.error("❌ API is offline")
        st.warning("Make sure the FastAPI server is running.")
        st.stop()
    
    # Client selection
    st.header("Client Selection")
    client_id = st.selectbox(
        "Select a client",
        options=DEFAULT_CLIENTS,
        index=0 if not st.session_state.current_client else DEFAULT_CLIENTS.index(st.session_state.current_client)
    )
    
    if client_id != st.session_state.current_client:
        st.session_state.current_client = client_id
        st.session_state.documents = list_documents(client_id)
        st.rerun()
    
    # File upload
    st.header("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "csv", "txt"],
        help="Upload a document for the selected client."
    )
    
    if uploaded_file:
        # Store the file name to check if it was already processed
        file_name = uploaded_file.name
        
        # Check if this file is already being processed
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != file_name:
            with st.spinner(f"Uploading file {file_name}..."):
                response = upload_file(uploaded_file, client_id)
                if response:
                    st.success(f"File uploaded successfully: {file_name}")
                    # Store the uploaded file name to prevent reprocessing
                    st.session_state.last_uploaded_file = file_name
                    # Update documents list
                    st.session_state.documents = list_documents(client_id)
                    # Use a placeholder instead of rerun to show fresh data
                    st.empty()
        else:
            st.info(f"File {file_name} has already been uploaded. Select a different file or refresh the page to upload again.")
    
    # Document list
    st.header("Client Documents")

    # Auto-refresh document status
    auto_refresh = auto_refresh_documents()
    if auto_refresh:
        st.empty()  # Force a subtle UI update

    # Add manual refresh button for document status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Documents are automatically refreshed every 15 seconds")
    with col2:
        if st.button("🔄 Refresh Now"):
            st.session_state.documents = list_documents(client_id)
            st.session_state.last_poll_time = datetime.now()
            st.success("Document status refreshed!")

    # Track if there are ready documents
    has_ready_documents = False

    if st.session_state.documents:
        for doc in st.session_state.documents:
            try:
                # Check if doc is a dictionary
                if isinstance(doc, dict):
                    status = doc.get("status", "unknown")
                    file_name = doc.get("name", doc.get("file_name", "Unknown"))
                    created_at = doc.get("created_at", "Unknown")
                # Check if doc is a string (might be a JSON string)
                elif isinstance(doc, str):
                    st.warning("Document data returned in unexpected format. Contact administrator.")
                    # Try to parse as JSON if it might be a JSON string
                    try:
                        doc_dict = json.loads(doc)
                        status = doc_dict.get("status", "unknown")
                        file_name = doc_dict.get("file_name", "Unknown")
                        created_at = doc_dict.get("created_at", "Unknown")
                    except json.JSONDecodeError:
                        # If not JSON, just use the string as the file name
                        status = "unknown"
                        file_name = doc
                        created_at = "Unknown"
                else:
                    # Fallback for any other type
                    status = "unknown"
                    file_name = str(doc)
                    created_at = "Unknown"
                    st.warning(f"Unexpected document type: {type(doc)}")
                    
                # Get status emoji and progress percentage
                status_emoji = "✅" if status == "ready" else "⏳" if status in ["pending", "partitioning", "chunked", "indexed"] else "❌"
                progress_percentage = get_progress_percentage(status)
                
                # Track if we have any ready documents
                if status == "ready":
                    has_ready_documents = True
                
                # Show more detailed status information
                st.markdown(f"{status_emoji} **{file_name}**")
                
                # Show progress bar
                if status != "ready" and status != "failed":
                    st.progress(progress_percentage / 100)
                    st.caption(f"Processing: {progress_percentage}% complete - Status: {status}")
                    st.caption("_Document is still processing and not yet available for querying_")
                elif status == "ready":
                    st.success("Document is ready for querying")
                elif status == "failed":
                    st.error("Processing failed")
                
                st.caption(f"Uploaded: {created_at}")
                st.markdown("---")
            except Exception as e:
                st.error(f"Error displaying document: {e}")
                st.text(f"Raw data: {doc}")
    else:
        st.info("No documents found for this client.")
    
    # Add a note about document processing
    st.markdown("""
    #### Note:
    Documents must be in **ready** status ✅ before they can be queried.
    Processing large files can take several minutes.
    """)

# Main area for chat interface
st.title("💬 Chat with Your Documents")

# Display instructions
st.markdown("""
Ask questions about your uploaded documents. The system will retrieve relevant information and provide answers with citations.
""")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        display_chat_message(message["content"], message["is_user"])
        if message.get("citations"):
            display_citations(message["citations"])

# Input area
query = st.chat_input("Ask a question about your documents..." if has_ready_documents else "Waiting for documents to be ready...")
if not has_ready_documents:
    st.warning("⚠️ No documents are ready for querying yet. Please wait for documents to reach 'ready' status before asking questions.")

if query and has_ready_documents:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "content": query,
        "is_user": True,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with chat_container:
        display_chat_message(query, True)
    
    # Get response from API
    with st.spinner("Thinking..."):
        response = query_documents(query, client_id)
        
        if response:
            answer = response.get("answer", "Sorry, I couldn't find an answer to your question.")
            citations = response.get("citations", [])
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "content": answer,
                "is_user": False,
                "timestamp": datetime.now().isoformat(),
                "citations": citations
            })
            
            # Display assistant message
            with chat_container:
                display_chat_message(answer, False)
                if citations:
                    display_citations(citations)
        else:
            # Display error message
            with chat_container:
                display_chat_message("Sorry, I encountered an error while processing your query. Please try again.", False)

# Footer
st.markdown("---")
st.caption("Ragie RAG Agent - Powered by OpenAI and Ragie") 