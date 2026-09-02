"""
Local development server for MedCite.

    python run_local.py

This lives outside app.py on purpose. Vercel runs `python app.py` during the
build to find the WSGI `app` object, so an `app.run()` inside that file would
start a server the build waits on forever. Keeping the server here means
app.py only ever defines things.

Serving on Vercel does not use this file at all.
"""

import os

from app import app

if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8080)),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )
