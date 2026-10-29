"""Build a Pod start command with a boot watchdog (RUNPOD.md Section 3; E-42 incident; E-44 A1 item 5).

The watchdog script is embedded as base64, so the command needs no nested JSON quoting. At boot the command writes
the script to /boot_watchdog.py, starts it in the background (it stops this Pod with GraphQL podStop after CAP
seconds, using the Pod-scoped RUNPOD_API_KEY and RUNPOD_POD_ID that Runpod sets in the container), then installs and
starts sshd as in E-36 to E-42.

Usage:
  pod_cmd.py --cap 4800                 print the "cmd" list as JSON (for create-pod)
  pod_cmd.py --cap 180 --script         print only the watchdog script (for tests)
"""

from __future__ import annotations

import argparse
import base64
import json

WATCHDOG = r'''
import json, os, sys, time, urllib.request
cap = int(sys.argv[1])
log = open(sys.argv[2] if len(sys.argv) > 2 else "/workspace/boot_watchdog.log", "a")
print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "armed cap", cap, file=log, flush=True)
time.sleep(cap)
url = os.environ.get("BOOT_WATCHDOG_URL", "https://api.runpod.io/graphql")
q = {"query": 'mutation { podStop(input: {podId: "%s"}) { id } }' % os.environ.get("RUNPOD_POD_ID", "")}
req = urllib.request.Request(url, data=json.dumps(q).encode(), method="POST", headers={
    "Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("RUNPOD_API_KEY", "")})
for attempt in range(5):
    try:
        status = urllib.request.urlopen(req, timeout=30).status
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "podStop", status, file=log, flush=True)
        if status == 200:
            break
    except Exception as e:
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "podStop error", type(e).__name__, e, file=log, flush=True)
    time.sleep(30)
'''

SSHD = ("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync && "
        "mkdir -p /root/.ssh /run/sshd && echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && "
        "chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity")


def command(cap: int) -> str:
    b64 = base64.b64encode(WATCHDOG.encode()).decode()
    return (f"mkdir -p /workspace && echo {b64} | base64 -d > /boot_watchdog.py && "
            f"(nohup python3 /boot_watchdog.py {cap} /workspace/boot_watchdog.log > /dev/null 2>&1 &) && " + SSHD)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cap", type=int, required=True, help="seconds from boot until the Pod stops itself")
    p.add_argument("--script", action="store_true")
    a = p.parse_args()
    print(WATCHDOG if a.script else json.dumps([command(a.cap)]))


if __name__ == "__main__":
    main()
