import os
import sys
import subprocess

# Simple handler to keep Vercel happy
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b"<html><head><meta http-equiv='refresh' content='5'></head><body style='background-color:#FFF0F5; font-family:sans-serif; text-align:center; padding-top:50px;'><h2 style='color:#D87093;'>Waking up the FL Engine...</h2><p>Please wait, loading hardware data.</p></body></html>"]

if __name__ == "__main__":
    # Launching dashboard.py with optimizations
    subprocess.Popen([
        "streamlit", "run", "dashboard.py",
        "--server.port", os.getenv("PORT", "8080"),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--theme.primaryColor", "#D87093"
    ])
