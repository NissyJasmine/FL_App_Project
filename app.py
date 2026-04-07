from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import subprocess
import os

app = FastAPI()

@app.get("/get-devices")
def get_devices():
    try:
        df = pd.read_excel("devices_dataset.csv.xlsx")
        return {"status": "success", "data": df.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# This special block triggers Streamlit when the server boots up
if __name__ == "__main__":
    subprocess.Popen([
        "streamlit", "run", "dashboard.py", 
        "--server.port", "8080", 
        "--server.address", "0.0.0.0"
    ])
