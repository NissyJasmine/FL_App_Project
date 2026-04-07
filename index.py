import os
import sys
from streamlit.web.cli import main

if __name__ == "__main__":
    # This tells Streamlit to run your dashboard code 
    # on the specific port Vercel expects
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
