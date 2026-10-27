"""Modular runner for the Ranked Extractive QA Web Application.

Runs the Flask QA web application.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "app")):
    if path not in sys.path:
        sys.path.insert(0, path)

from qa_web import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("QA_HOST", "127.0.0.1")
    port = int(os.environ.get("QA_PORT", "5000"))
    debug = os.environ.get("QA_DEBUG", "0") == "1"

    print("=" * 60)
    print("  Ranked Extractive QA Web Application")
    print("=" * 60)
    print(f"  * Host / Port  : http://{host}:{port}")
    print(f"  * Environment  : {app.config_object.__name__}")
    print(f"  * Active Model : {app.config_object.MODEL_DIR}")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug, use_reloader=False)
