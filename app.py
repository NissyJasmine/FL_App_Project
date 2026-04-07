import os
import subprocess
from fastapi import FastAPI

app = FastAPI()

@app.get("/get-devices")
def get_devices():
    # Keep your existing device data logic here
    return {"status": "online", "data": []} 

# This is the "Magic" part for Vercel
if __name__ == "__main__":
    # This command manually triggers Streamlit on the port Vercel expects
    subprocess.run([
        "streamlit", 
        "run", 
        "dashboard.py", 
        "--server.port", "8080", 
        "--server.address", "0.0.0.0"
    ])
