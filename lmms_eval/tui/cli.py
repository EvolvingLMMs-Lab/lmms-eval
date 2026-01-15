#!/usr/bin/env python3
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    import urllib.error
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.3)
    return False


def check_web_built() -> bool:
    dist_dir = Path(__file__).parent / "web" / "dist"
    return (dist_dir / "index.html").exists()


def build_web_ui() -> bool:
    web_dir = Path(__file__).parent / "web"
    if not (web_dir / "package.json").exists():
        print("Web UI source not found", file=sys.stderr)
        return False

    print("Building web UI...")

    if not (web_dir / "node_modules").exists():
        result = subprocess.run(["npm", "install"], cwd=web_dir)
        if result.returncode != 0:
            print("Failed to install dependencies", file=sys.stderr)
            return False

    result = subprocess.run(["npm", "run", "build"], cwd=web_dir)
    if result.returncode != 0:
        print("Failed to build web UI", file=sys.stderr)
        return False

    return True


def main() -> int:
    port = int(os.environ.get("LMMS_SERVER_PORT", "8000"))
    server_url = f"http://localhost:{port}"

    if not check_web_built():
        if not build_web_ui():
            return 1

    print(f"Starting LMMs-Eval Web UI on {server_url}")

    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "lmms_eval.tui.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def cleanup():
        if server_process and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_process.kill()

    atexit.register(cleanup)

    if not wait_for_server(server_url):
        print("Failed to start server", file=sys.stderr)
        return 1

    print(f"Server running at {server_url}")
    print("Opening browser...")
    webbrowser.open(server_url)

    print("Press Ctrl+C to stop")

    try:
        while True:
            if server_process.poll() is not None:
                print("Server stopped unexpectedly", file=sys.stderr)
                return 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        return 0


if __name__ == "__main__":
    sys.exit(main())
