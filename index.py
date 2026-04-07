import os
import sys
from streamlit.web.cli import main

if __name__ == "__main__":
    # This tells Vercel to launch your dashboard.py UI
    sys.argv = [
        "streamlit",
        "run",
        "dashboard.py",
        "--server.port",
        os.getenv("PORT", "8080"),
        "--server.address",
        "0.0.0.0",
    ]
    sys.exit(main())
