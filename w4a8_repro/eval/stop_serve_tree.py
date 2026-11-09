#!/usr/bin/env python3
"""Stop ONLY this run's serving tree — never other services in the container.

The original 10007 stop_server.py sweeps every `VLLM::` process in its
container; that is unsafe here because the 20002 wkp-vllm container may host
other services (e.g. the colleague's serving on cards 0-3). This port instead
identifies victims precisely; a process is ours iff ANY of:
  - its process group id equals <pgid> (the setsid'd server and its children)
  - our unique bootstrap dir is in its environment (inherited PYTHONPATH)
  - one of the cmdline markers matches (model path / launcher script)

Our own ancestors (the calling shell and up) are always excluded.

Usage: python3 stop_serve_tree.py <pgid> [cmdline_marker ...]
"""
import os
import signal
import sys
import time
from pathlib import Path

BOOT_MARKER = "/home/w30022218/w4a8_tuning/ceval/bootstrap"


def _env_has_exact_bootstrap(env: str) -> bool:
    """True only if PYTHONPATH contains BOOT_MARKER as an exact entry.

    A plain substring test makes '.../bootstrap' also match
    '.../bootstrap_main' (and vice versa), cross-killing concurrent runs.
    """
    for line in env.split("\n"):
        if line.startswith("PYTHONPATH="):
            entries = line[len("PYTHONPATH="):].split(":")
            return any(e == BOOT_MARKER for e in entries)
    return False


def ancestors():
    """PIDs of self and all ancestors (calling shell, ssh chain, ...)."""
    out = set()
    pid = os.getpid()
    while pid > 1:
        out.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return out


def victims(pgid, markers):
    protected = ancestors()
    out = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in protected:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if not cmd:
            continue
        try:
            env = (entry / "environ").read_bytes().decode("utf-8", "replace")
        except OSError:
            env = ""
        mine = False
        try:
            mine = pgid != 0 and os.getpgid(pid) == pgid
        except OSError:
            pass
        # pgid is authoritative for setsid'd trees; the env criterion is a
        # fallback ONLY when the server pgid is unknown, otherwise concurrent
        # runs sharing one bootstrap dir would cross-kill each other.
        if mine or (pgid == 0 and _env_has_exact_bootstrap(env)) or any(m and m in cmd for m in markers):
            out.add(pid)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    pgid = int(sys.argv[1])
    markers = sys.argv[2:]
    for sig, wait in ((signal.SIGTERM, 6), (signal.SIGKILL, 3)):
        pids = victims(pgid, markers)
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        time.sleep(wait)
    remaining = victims(pgid, markers)
    print("stop_serve_tree: remaining:", sorted(remaining) if remaining else "none")
    sys.exit(1 if remaining else 0)


if __name__ == "__main__":
    main()
