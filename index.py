import os
import sys
import subprocess
import time

# This sends a signal to the browser to refresh once the engine is ready
def app(environ, start_response):
    status = '200 OK'
    headers = [('Content-Type', 'text/html')]
    start_response(status, headers)
    return [b"<html><head><meta http-equiv='refresh' content='10'></head><body><h2 style='color:#D87093; font-family:sans-serif;'>Federated Learning Engine is Warming Up...</h2><p>The dashboard will load automatically in a few seconds.</p></body></html>"]

if __name__ == "__main__":
    # Standard headless command for cloud deployment
    subprocess.Popen([
        "streamlit", "run", "dashboard.py",
        "--server.port", os.getenv("PORT", "8080"),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])
