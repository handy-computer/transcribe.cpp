from __future__ import annotations

from collections import deque
import subprocess
import sys
import threading


def run_subprocess_capturing_stderr(
    cmd: list[str],
    cwd: str,
    env: dict,
) -> tuple[int, str]:
    """Run cmd with stdout inheriting (live progress to Modal logs) and stderr
    tee'd through a background thread so we both stream and capture it.

    Returns (returncode, stderr_tail) where stderr_tail is up to the last 80
    lines, suitable for inclusion in a failure exception.
    """
    tail: deque[str] = deque(maxlen=80)

    def _tee(stream):
        for line in stream:
            sys.stderr.write(line)
            sys.stderr.flush()
            tail.append(line)

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    thr = threading.Thread(target=_tee, args=(proc.stderr,), daemon=True)
    thr.start()
    rc = proc.wait()
    thr.join(timeout=5)
    return rc, "".join(tail)
