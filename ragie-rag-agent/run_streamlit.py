"""
Run script for the Streamlit frontend.
"""
import os
import subprocess
import sys

def run_streamlit():
    """
    Run the Streamlit app.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the path to the Streamlit app
    streamlit_app_path = os.path.join(current_dir, "streamlit_app.py")
    
    # Run Streamlit
    subprocess.run([
        "streamlit", "run", 
        streamlit_app_path,
        "--server.port", "8501",
        "--browser.serverAddress", "localhost",
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    run_streamlit() 