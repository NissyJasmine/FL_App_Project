import os
import subprocess
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd

app = FastAPI()

# 1. THE DATA API (Keep this for your background data)
@app.get("/get-devices")
def get_devices():
    try:
        df = pd.read_excel("devices_dataset.csv.xlsx")
        return {"status": "success", "data": df.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 2. THE MAIN INTERFACE
@app.get("/", response_class=HTMLResponse)
def root():
    # This forces Vercel to realize this isn't just a text API
    return """
    <html>
        <head><title>FL Device Selection</title></head>
        <body style="background-color: #ffe4e1; display: flex; justify-content: center; align-items: center; height: 100vh; font-family: sans-serif;">
            <div style="text-align: center;">
                <h1>Launching Pink Dashboard...</h1>
                <p>If it doesn't load, <a href="/get-devices">check API Data here</a>.</p>
            </div>
        </body>
    </html>
    """

# 3. THE STREAMLIT COMMAND
# This runs when Vercel boots the function
if __name__ == "__main__":
    subprocess.Popen(["streamlit", "run", "dashboard.py", "--server.port", "8080", "--server.address", "0.0.0.0"])
