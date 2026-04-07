from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import subprocess
import os
import sys

app = FastAPI()

# --- CONFIGURATION ---
DATA_FILE = "devices_dataset.csv.xlsx"

# --- API ENDPOINTS ---

@app.get("/get-devices")
def get_devices():
    """Returns the device data as JSON for the dashboard to consume."""
    try:
        if not os.path.exists(DATA_FILE):
            return JSONResponse(
                status_code=404, 
                content={"error": f"Dataset file {DATA_FILE} not found."}
            )
        
        # Load the data
        df = pd.read_excel(DATA_FILE)
        # Convert to dictionary for JSON response
        data = df.to_dict(orient="records")
        return {"status": "success", "data": data}
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e)}
        )

@app.get("/")
def home():
    """
    Main landing page. On Vercel, we use this to verify the API 
    is live before the Dashboard takes over routing.
    """
    return {
        "status": "online",
        "message": "Adaptive Client Selection API is running",
        "endpoints": {
            "get_devices": "/get-devices",
            "docs": "/docs"
        }
    }

# --- VERCEL SURVIVAL LOGIC ---
# This part is critical for running Streamlit within a Vercel function environment.
if __name__ == "__main__":
    # If the script is run directly, it starts the Streamlit dashboard
    # on port 8080 which Vercel listens to.
    subprocess.run([
        "streamlit", 
        "run", 
        "dashboard.py", 
        "--server.port", "8080", 
        "--server.address", "0.0.0.0"
    ])
