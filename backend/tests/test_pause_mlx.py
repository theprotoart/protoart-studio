"""Pause on the fast engine suspends its worker process and Resume wakes it (no MLX needed)."""
import subprocess
import sys
import threading
import unittest
from pathlib import Path

import mlx_engine


def state(pid):
    p = Path(f"/proc/{pid}/stat")
    return p.read_text().split(")")[-1].split()[0] if p.exists() else None


@unittest.skipUnless(Path("/proc/self/stat").exists(), "needs /proc")
class FreezeTests(unittest.TestCase):
    def test_freeze_and_thaw(self):
        e = mlx_engine.MlxEngine(lambda m: None, threading.Event(), threading.Event())
        e.proc = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(0.01)"])
        try:
            e._freeze(True)
            self.assertTrue(e.frozen)
            for _ in range(50):
                if state(e.proc.pid) == "T":
                    break
                import time; time.sleep(0.02)
            self.assertEqual(state(e.proc.pid), "T")
            e._freeze(False)
            self.assertFalse(e.frozen)
            import time; time.sleep(0.1)
            self.assertNotEqual(state(e.proc.pid), "T")
        finally:
            e.proc.kill()
            e.proc.wait()
        e._freeze(True)  # process gone: nothing to do, no error
        self.assertFalse(e.frozen)


if __name__ == "__main__":
    unittest.main()
