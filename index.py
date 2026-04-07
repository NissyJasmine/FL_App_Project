import os
import sys
import subprocess

# satisfies Vercel's requirement for a top-level handler
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b"Streamlit engine is warming up... Please refresh in 30 seconds."]

if __name__ == "__main__":
    # Get the current directory to ensure devices_dataset is found
    base_path = os.path.dirname(__file__)
    dashboard_path = os.path.join(base_path, "dashboard.py")
    
    subprocess.Popen([
        "streamlit", "run", dashboard_path,
        "--server.port", os.getenv("PORT", "8080"),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])
