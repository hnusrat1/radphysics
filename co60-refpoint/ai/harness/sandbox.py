"""Isolated Python sandbox for conditions A1 and A2 (STUDY_PLAN Section 5.2).

Model-written code runs in fresh Linux namespaces:
  network  no interfaces except loopback down: no internet, no DNS
  mount    the study repository, the transport codes (Geant4, EGSnrc, their builds and sources), the queue
           logs, the session transcripts and /tmp are hidden under empty tmpfs mounts, so no Monte Carlo
           result and no transport code is reachable
  pid      the code sees only its own processes (no /proc of the harness or of running simulations)
and with fixed limits: 4 cores (taskset), CPU time and wall time per call, address space, file size, and an
empty environment (no keys, no proxy settings). The working directory persists across the calls of one
conversation so the model can keep files; for A2 it holds read-only copies of the data tables.

Requires root (unshare -m). The Python is the study venv (numpy, scipy); nothing else is installed for it.
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

PYTHON = os.environ.get("SANDBOX_PYTHON", "/opt/venv/bin/python")
HIDDEN = ["/home", "/root", "/opt/EGSnrc", "/opt/geant4", "/opt/build", "/opt/src", "/opt/queue",
          "/opt/claude-code", "/opt/env-runner", "/tmp"]
LIMITS = {"cores": 4, "cpu_s": 4 * 600, "wall_s": 600, "address_space_bytes": 16 * 2**30,
          "file_size_bytes": 512 * 2**20, "output_chars": 20000}


class Sandbox:
    """One per conversation: a working directory outside every hidden path, seen as /tmp/work inside
    (data files at /tmp/data)."""

    def __init__(self, data_files=(), root="/var/tmp/co60-sandbox"):
        Path(root).mkdir(parents=True, exist_ok=True)
        Path("/var/lib/co60-sandbox-mnt").mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix="conv-", dir=root))
        self.data = self.dir / "data"
        self.work = self.dir / "work"
        self.data.mkdir()
        self.work.mkdir()
        for f in data_files:
            dst = self.data / Path(f).name
            shutil.copyfile(f, dst)
            dst.chmod(0o444)
        self.calls = 0

    def run(self, code, limits=LIMITS):
        self.calls += 1
        script = self.work / f"_call_{self.calls:03d}.py"
        script.write_text(code)
        mounts = " && ".join(f"mount -t tmpfs -o size=64m none {d}" for d in HIDDEN if Path(d).exists())
        # Bind this conversation's directory onto /tmp first, then hide /var/tmp (other conversations).
        inner = (f"mount --bind {self.dir} /var/lib/co60-sandbox-mnt && {mounts} && "
                 f"mount --bind /var/lib/co60-sandbox-mnt /tmp && umount /var/lib/co60-sandbox-mnt && "
                 f"mount -t tmpfs -o size=1m none /var/tmp && cd /tmp/work && "
                 f"exec prlimit --cpu={limits['cpu_s']} --as={limits['address_space_bytes']} "
                 f"--fsize={limits['file_size_bytes']} --nproc=256 "
                 f"taskset -c 0-{limits['cores'] - 1} {PYTHON} -I {script.name}")
        # The working directory is bound to /tmp inside, after /tmp itself was hidden; data at /tmp/data.
        cmd = ["unshare", "-n", "-m", "-p", "-f", "--mount-proc", "--propagation", "private", "--",
               "sh", "-c", inner]
        env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp/work", "LANG": "C.UTF-8", "OMP_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4", "MPLBACKEND": "Agg"}
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=limits["wall_s"], env=env)
            out, err, code_ = p.stdout, p.stderr, p.returncode
            timed_out = False
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            code_, timed_out = None, True
        wall = time.time() - t0
        n = limits["output_chars"]
        return {"stdout": out[-n:], "stderr": err[-n:], "exit_code": code_, "timed_out": timed_out,
                "wall_s": round(wall, 2), "stdout_truncated": len(out) > n, "stderr_truncated": len(err) > n}

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)
