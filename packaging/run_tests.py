"""Runs every backend self-test (tests/test_*.py): plain test_ functions and unittest classes.
Prints one line per test and a summary; exits 1 if any failed. Used by "Rebuild installer.command"."""
import importlib
import inspect
import os
import pathlib
import sys
import tempfile
import time
import unittest

BK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, BK)
os.chdir(BK)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download anything while testing

ok, failed = 0, []
start = time.time()
for f in sorted(pathlib.Path(BK, "tests").glob("test_*.py")):
    mod = "tests." + f.stem
    try:
        T = importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        failed.append(f"{f.stem} (couldn't load: {type(e).__name__}: {e})")
        print("FAIL", f.stem, "couldn't load:", e, flush=True)
        continue
    for name, fn in inspect.getmembers(T, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != T.__name__:
            continue
        try:
            fn(pathlib.Path(tempfile.mkdtemp())) if "tmp_path" in inspect.signature(fn).parameters else fn()
            ok += 1
        except unittest.SkipTest:
            pass
        except Exception as e:  # noqa: BLE001
            failed.append(f"{f.stem}.{name}: {type(e).__name__}: {e}")
            print("FAIL", f.stem, name, type(e).__name__, e, flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromModule(T)
    if suite.countTestCases():
        res = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
        ok += res.testsRun - len(res.failures) - len(res.errors) - len(res.skipped)
        for case, tb in res.failures + res.errors:
            last = tb.strip().splitlines()[-1]
            failed.append(f"{case.id()}: {last}")
            print("FAIL", case.id(), last, flush=True)
    print(f"  {f.stem}: done", flush=True)

print(f"\n{ok} tests passed, {len(failed)} failed ({time.time() - start:.0f} s)")
sys.exit(1 if failed else 0)
