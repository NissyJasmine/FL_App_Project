import os
import subprocess
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI()

# --- 1. DATA API LOGIC ---
@app.get("/get-devices")
def get_devices():
    try:
        df = pd.read_excel("devices_dataset.csv.xlsx")
        return {"status": "success", "data": df.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 2. THE DASHBOARD LAUNCHER ---
# This is what forces the Pink Dashboard to show up on the main page
if __name__ == "__main__":
    import uvicorn
    # This runs Streamlit as a subprocess on the port Vercel expects
    subprocess.Popen([
        "streamlit", "run", "dashboard.py", 
        "--server.port", "8080", 
        "--server.address", "0.0.0.0"
    ])
    # This keeps the FastAPI backend alive in the background
    uvicorn.run(app, host="0.0.0.0", port=8000)
