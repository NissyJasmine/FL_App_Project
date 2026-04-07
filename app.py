import os
import subprocess
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd

app = FastAPI()

@app.get("/get-devices")
def get_devices():
    """The API endpoint for your data."""
    try:
        df = pd.read_excel("devices_dataset.csv.xlsx")
        return {"status": "success", "data": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def home():
    """The landing page that triggers the UI."""
    return """
    <html>
        <head><title>Launching Dashboard</title></head>
        <body style="background-color: #ffe4e1; font-family: Arial; text-align: center; padding-top: 50px;">
            <h1>Connecting to Federated Learning Dashboard...</h1>
            <p>If the dashboard doesn't load in 5 seconds, please refresh.</p>
            <script>
                // This script will attempt to trigger the UI logic
                console.log("Dashboard initializing...");
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    # This line is what Vercel uses to boot your Streamlit UI
    subprocess.Popen([
        "streamlit", "run", "dashboard.py", 
        "--server.port", "8080", 
        "--server.address", "0.0.0.0"
    ])
