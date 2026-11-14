"""The version the window shows (in Copy diagnostics and Settings) is the app's real version (app/package.json)."""
import json
import re
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"


class VersionInStep(unittest.TestCase):
    def test_window_shows_the_real_version(self):
        real = json.loads((APP / "package.json").read_text())["version"]
        shown = re.search(r'const APP_VERSION = "([^"]+)"', (APP / "renderer" / "app.js").read_text()).group(1)
        self.assertEqual(shown, real, "update APP_VERSION in app/renderer/app.js to match app/package.json")


if __name__ == "__main__":
    unittest.main()
