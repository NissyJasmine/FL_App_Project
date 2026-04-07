import os
import sys
import subprocess

# This 'app' object satisfies Vercel's requirement for a top-level handler
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b"Streamlit is starting..."]

if __name__ == "__main__":
    # Launch the Streamlit dashboard
    subprocess.Popen([
        "streamlit", "run", "dashboard.py",
        "--server.port", os.getenv("PORT", "8080"),
        "--server.address", "0.0.0.0"
    ])
