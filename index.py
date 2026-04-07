import os
import sys
import subprocess

# satisfies Vercel's requirement for a top-level handler
def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b"Streamlit engine is warming up... Please refresh in 30 seconds."]

if __name__ == "__main__":
    subprocess.Popen([
        "streamlit", "run", "dashboard.py",
        "--server.port", os.getenv("PORT", "8080"),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])
