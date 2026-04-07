import os
import sys
from streamlit.web.cli import main

if __name__ == "__main__":
    # Force Streamlit to run on the port Vercel provides
    sys.argv = [
        "streamlit",
        "run",
        "dashboard.py",
        "--server.port",
        "8080",
        "--server.address",
        "0.0.0.0",
    ]
    sys.exit(main())
