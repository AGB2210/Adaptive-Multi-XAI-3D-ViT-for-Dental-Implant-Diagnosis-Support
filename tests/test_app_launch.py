"""Starting and stopping the app: the port it takes, and what `start.bat` needs.

`start.bat` is one click because `python -m app` decides these things itself --
whether the port is free, whether the app is already running there, whether a
package is missing -- rather than failing on a bind error and a traceback.
"""

from __future__ import annotations

import http.server
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from app import __main__ as launcher  # noqa: E402
from app.server import APP_ID  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


@pytest.fixture
def listener():
    """Some other program holding a port: it accepts connections and says nothing."""
    sock = socket.socket()
    sock.bind((HOST, 0))
    sock.listen()
    yield sock.getsockname()[1]
    sock.close()


@pytest.fixture
def serving():
    """A server answering /api/status with whatever body the test hands it."""
    body = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer((HOST, 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], body
    server.shutdown()
    server.server_close()


class TestThePort:
    def test_a_free_port_is_taken_as_configured(self):
        port = free_port()
        assert launcher.choose_port(HOST, port, explicit=False) == (port, None)

    def test_a_port_someone_is_listening_on_is_not_free(self, listener):
        assert not launcher.port_is_free(HOST, listener)

    def test_the_next_free_port_is_used_when_another_program_holds_it(self, listener, capsys):
        port, already = launcher.choose_port(HOST, listener, explicit=False)
        assert already is None and port != listener
        assert listener < port <= listener + launcher.PORT_SEARCH
        assert launcher.port_is_free(HOST, port)
        assert str(listener) in capsys.readouterr().out, "a changed port must be announced"

    def test_a_port_asked_for_by_name_is_never_quietly_changed(self, listener):
        with pytest.raises(SystemExit, match=str(listener)):
            launcher.choose_port(HOST, listener, explicit=True)

    def test_the_app_already_running_there_is_recognised(self, serving):
        port, body = serving
        body.update({"app": APP_ID, "version": "9.9.9"})
        for explicit in (False, True):
            assert launcher.choose_port(HOST, port, explicit=explicit) == (port, body)

    def test_another_web_server_is_not_mistaken_for_the_app(self, serving):
        port, body = serving
        body.update({"app": "something-else"})
        assert launcher.running_instance(HOST, port) is None
        chosen, already = launcher.choose_port(HOST, port, explicit=False)
        assert already is None and chosen != port

    def test_a_wildcard_bind_is_opened_on_loopback(self):
        assert launcher.browser_host("0.0.0.0") == "127.0.0.1"
        assert launcher.browser_host("127.0.0.1") == "127.0.0.1"
        assert launcher.browser_host("::1") == "[::1]"


class TestMissingPackages:
    def test_a_missing_package_exits_with_the_code_the_launcher_installs_on(
            self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "uvicorn", None)       # import uvicorn -> not found
        monkeypatch.setattr(sys, "argv", ["app"])
        with pytest.raises(SystemExit) as stop:
            launcher.main()
        assert stop.value.code == launcher.EXIT_MISSING_PACKAGE
        out = capsys.readouterr().out
        assert "uvicorn" in out and "requirements-app.txt" in out

    def test_start_bat_installs_on_that_same_code(self):
        text = (REPO / "start.bat").read_text(encoding="ascii")
        assert f'if "%CODE%"=="{launcher.EXIT_MISSING_PACKAGE}"' in text


class TestStartBat:
    def test_it_has_windows_line_endings(self):
        """cmd mis-parses labels and `goto` in a batch file with LF endings, and
        the repository normalises every other file to LF."""
        data = (REPO / "start.bat").read_bytes()
        assert data.count(b"\r\n") == data.count(b"\n") > 0
        assert "*.bat text eol=crlf" in (REPO / ".gitattributes").read_text(encoding="utf-8")

    def test_every_goto_has_its_label(self):
        text = (REPO / "start.bat").read_text(encoding="ascii")
        lines = [line.strip() for line in text.splitlines()]
        labels = {line[1:].split()[0] for line in lines if line.startswith(":")}
        targets = {word[1:] for line in lines if not line.lower().startswith("rem")
                   for a, word in zip(line.split(), line.split()[1:])
                   if a.lower() in ("goto", "call") and word.startswith(":")}
        assert targets and targets <= labels, targets - labels
