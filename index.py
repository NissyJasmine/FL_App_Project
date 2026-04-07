import os
import subprocess

def app(environ, start_response):
    # This creates a professional loading screen while Streamlit starts
    start_response('200 OK', [('Content-Type', 'text/html')])
    html = """
    <html>
    <head>
        <meta http-equiv='refresh' content='8'>
        <style>
            body { background-color: #FFF0F5; font-family: sans-serif; text-align: center; padding-top: 100px; }
            .loader { border: 8px solid #f3f3f3; border-top: 8px solid #D87093; border-radius: 50%; width: 50px; height: 50px; animation: spin 2s linear infinite; margin: auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            h2 { color: #D87093; }
        </style>
    </head>
    <body>
        <div class="loader"></div>
        <h2>Final Project: FL Device Selection Engine</h2>
        <p>The dashboard is initializing on Vercel. <br><b>Please wait 30 seconds...</b></p>
    </body>
    </html>
    """
    return [html.encode('utf-8')]

if __name__ == "__main__":
    subprocess.Popen([
        "streamlit", "run", "dashboard.py",
        "--server.port", os.getenv("PORT", "8080"),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])
