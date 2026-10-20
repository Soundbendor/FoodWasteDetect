"""Exercise cache cleanup without downloading or importing a model."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from internvl.scripts.smoke_jina import _run_worker


class SmokeCleanupTests(unittest.TestCase):
    def test_success_and_failure_remove_cache(self):
        for status in (0, 7):
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
            ):
                record = Path(directory) / "cache-path"
                code = (
                    "import os, pathlib, sys; "
                    f"pathlib.Path({str(record)!r}).write_text(os.environ['HF_HOME']); "
                    f"sys.exit({status})"
                )
                self.assertEqual(_run_worker([sys.executable, "-c", code]), status)
                self.assertFalse(Path(record.read_text()).parent.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX signals")
    def test_interrupt_and_termination_stop_worker_and_remove_cache(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as directory:
                record = Path(directory) / "worker"
                child = (
                    "import os, pathlib, time; "
                    f"pathlib.Path({str(record)!r}).write_text("
                    "os.environ['HF_HOME'] + '\\n' + str(os.getpid())); time.sleep(60)"
                )
                parent = (
                    "from internvl.scripts.smoke_jina import _run_worker; "
                    f"_run_worker({[sys.executable, '-c', child]!r})"
                )
                process = subprocess.Popen([sys.executable, "-c", parent])
                try:
                    deadline = time.monotonic() + 10
                    while not record.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(record.exists(), "worker did not start")
                    cache, pid = record.read_text().splitlines()
                    process.send_signal(sig)
                    self.assertEqual(process.wait(timeout=15), 128 + sig)
                    self.assertFalse(Path(cache).parent.exists())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int(pid), 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
