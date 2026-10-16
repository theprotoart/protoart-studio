# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue:
use **GitHub → Security → Report a vulnerability** (private advisory) on this repository.

Include what you found, how to reproduce it, and the version (in the DMG's file name or `app/package.json`).
You should get a reply within a week. Please give us a reasonable time to fix it before disclosing it publicly.

## Supported versions

Only the latest release gets security fixes. The app has no automatic updates, so check the Releases page.

## How the app is built to be safe

- Everything runs on your Mac. The app has no account, no telemetry and no analytics.
- It only goes online to download a model or tool you asked for (Hugging Face, GitHub, PyPI). Settings → Privacy →
  Strict offline blocks even that.
- The window runs sandboxed with context isolation and no Node.js access; a narrow bridge (`app/preload.js`) is the
  page's only way to reach the Mac, and every bridge call is checked in `app/main.js`.
- Model files are loaded with safe loaders (safetensors first, `torch.load(weights_only=True)` for older formats);
  remote model code (`trust_remote_code`) is never run.
- Builds verify the checksum of the Python runtime they download, and use pinned dependency versions
  (`backend/requirements.lock.txt`).
- Each release publishes a SHA-256 checksum for its DMG.
