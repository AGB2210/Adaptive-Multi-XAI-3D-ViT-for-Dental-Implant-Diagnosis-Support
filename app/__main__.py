"""python -m app [--config configs/app.yaml] [--host H] [--port P] [--device D] [--open]

`start.bat` runs this. Three things here exist so that one click is enough and
stopping really stops:

  THE PORT   If the configured port already answers as this app, the page is
             opened on the instance that is running instead of failing on a
             bind error. If something else holds it, the next free port is
             taken and printed -- unless the port was asked for by name, in
             which case that request is refused rather than quietly changed.

  THE STOP   Ctrl+C shuts the server down and the port is released at once. A
             job that is mid-computation cannot be interrupted from outside its
             thread, and Python would otherwise wait for it before exiting --
             minutes, for a four-method explanation. The process is ended
             instead; every result is written through a temporary file and a
             rename, so nothing half-written is left, and the page offers the
             scan's analysis again.

  MISSING PACKAGES  exit with a code of their own, so the launcher can install
             the requirements and try again rather than show a traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

EXIT_MISSING_PACKAGE = 3
PORT_SEARCH = 50


def browser_host(host: str) -> str:
    """The address a browser on this machine reaches a server bound to `host` at."""
    return {"0.0.0.0": "127.0.0.1", "::": "[::1]"}.get(host, f"[{host}]" if ":" in host else host)


def port_is_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        # The question is "is anyone LISTENING here", and the two platforms need
        # opposite options to ask it.
        if os.name == "nt":
            # Without this a Windows bind can succeed on a port another process
            # is already listening on.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # Without this a bind fails for a minute or so after a server on the
            # port has stopped, while its closed connections linger. The server
            # sets it for its own bind, so it would have started; this check
            # said "in use by another program" and moved a restarted app to the
            # next port. Measured on Linux. A listening socket still refuses it.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def running_instance(host: str, port: int) -> dict | None:
    """/api/status of this app if it is what answers on the port, else None."""
    from app.server import APP_ID

    try:
        with urllib.request.urlopen(f"http://{browser_host(host)}:{port}/api/status",
                                    timeout=2) as res:
            status = json.loads(res.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return status if isinstance(status, dict) and status.get("app") == APP_ID else None


def choose_port(host: str, port: int, explicit: bool) -> tuple[int, dict | None]:
    """(port to serve on, status of an instance already serving there or None)."""
    if port_is_free(host, port):
        return port, None
    status = running_instance(host, port)
    if status is not None:
        return port, status
    if explicit:
        raise SystemExit(f"port {port} is in use by another program; pick another with --port")
    for candidate in range(port + 1, port + 1 + PORT_SEARCH):
        if port_is_free(host, candidate):
            print(f"Port {port} is in use by another program; using {candidate} instead.")
            return candidate, None
    raise SystemExit(f"no free port between {port} and {port + PORT_SEARCH}")


def open_when_ready(host: str, port: int, url: str) -> None:
    """Open the page once the server accepts a connection, from a daemon thread."""
    def wait_then_open():
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((browser_host(host).strip("[]"), port), timeout=1):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.2)

    threading.Thread(target=wait_then_open, name="open-browser", daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description="Implant site screening app")
    ap.add_argument("--config", default=None, help="defaults to configs/app.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--data-dir", dest="data_dir", default=None)
    ap.add_argument("--open", action="store_true", help="open the page in the browser")
    args = ap.parse_args()

    try:
        import uvicorn

        from app.server import create_app
        from app.settings import load_settings
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in ("app", "src"):
            raise                        # a fault in this code, not a missing install
        print(f"Missing Python package: {exc.name}\n"
              f"Install the app's requirements:  pip install -r requirements-app.txt")
        raise SystemExit(EXIT_MISSING_PACKAGE) from None

    settings = load_settings(args.config, host=args.host, port=args.port,
                             device=args.device, data_dir=args.data_dir)
    port, already = choose_port(settings.host, settings.port, explicit=args.port is not None)
    url = f"http://{browser_host(settings.host)}:{port}"
    if already is not None:
        print(f"Implant site screening is already running at {url} "
              f"(version {already.get('version')}). Stop that one to start another.")
        if args.open:
            webbrowser.open(url)
        return
    settings.port = port

    app = create_app(settings)
    print(f"Implant site screening -- {url}  (device {settings.device}, "
          f"data in {settings.data_dir})\nCtrl+C stops the server.", flush=True)
    if args.open:
        open_when_ready(settings.host, port, url)

    try:
        # Requests here take milliseconds; the grace period only bounds a
        # browser that holds its connection open while the server stops.
        uvicorn.run(app, host=settings.host, port=port, log_level="warning",
                    timeout_graceful_shutdown=3)
    except KeyboardInterrupt:
        pass                             # uvicorn re-raises the Ctrl+C it handled
    finally:
        running = app.state.jobs.shutdown()
        print("Server stopped." if not running else
              f"Server stopped; ending {running} job(s) that were still running. "
              f"Run them again from the page.", flush=True)
        if running:
            # A worker thread mid-forward-pass would hold the process open.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
