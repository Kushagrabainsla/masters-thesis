"""Tests for pod_cmd.py. Run: python3 -m pytest test_pod_cmd.py"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pod_cmd


def test_command_embeds_the_script_and_starts_sshd() -> None:
    cmd = json.loads(subprocess.check_output([sys.executable, str(Path(pod_cmd.__file__)), "--cap", "4800"]))[0]
    b64 = re.search(r"echo (\S+) \| base64 -d", cmd).group(1)
    assert base64.b64decode(b64).decode() == pod_cmd.WATCHDOG
    assert "/boot_watchdog.py 4800 " in cmd and cmd.endswith("sleep infinity") and "/usr/sbin/sshd" in cmd
    assert '"' not in cmd.split("&&")[1]  # no nested quotes in the watchdog part


def test_watchdog_posts_pod_stop_after_cap(tmp_path: Path) -> None:
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["auth"] = self.headers["Authorization"]
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    script = tmp_path / "wd.py"
    script.write_text(pod_cmd.WATCHDOG)
    env = dict(os.environ, RUNPOD_POD_ID="pod123", RUNPOD_API_KEY="k",
               BOOT_WATCHDOG_URL=f"http://127.0.0.1:{srv.server_port}/graphql")
    subprocess.run([sys.executable, str(script), "1", str(tmp_path / "wd.log")], env=env, timeout=30, check=True)
    assert seen["auth"] == "Bearer k" and 'podId: "pod123"' in seen["body"]["query"]
    assert "podStop 200" in (tmp_path / "wd.log").read_text()
