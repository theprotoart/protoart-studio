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
- The data folder (`~/.protoartstudio`) and its database are readable by your own Mac account only (saved Hugging Face
  and Civitai keys, prompts and picture paths are kept there). Saved keys are never sent back to the window, only
  whether one is saved, and each key only goes to its own site (huggingface.co; civitai.com / civitai.red).

## Audit: the "20 ways a vibe-coded app gets hacked" list (1.6.16)

Protoart Studio is a desktop app: no web server, no accounts, no passwords, no payments and no cloud database. Most of
the list is about web apps with those; each item was still checked against the code.

| # | Risk | Here | Result |
|---|------|------|--------|
| 1 | Next.js header bypass (`x-middleware-subrequest`) | No Next.js, no web server | Not applicable |
| 2 | Exposed `.git` folder | The app bundle copies named files only (`packaging/build_mac_app.sh`); the repository is public on purpose and its whole history scans clean with gitleaks | OK |
| 3 | Open Supabase storage | No Supabase or cloud storage | Not applicable |
| 4 | Price set by the browser | No payments | Not applicable |
| 5 | Credential stuffing | No accounts or logins | Not applicable |
| 6 | Open database | SQLite file on your Mac; nothing listens on a network port (the engine talks to the window through a private pipe) | **Fixed**: the folder is now owner-only (it used the default permissions) |
| 7 | Polyfill.io / third-party scripts | Every script is local; the page's CSP allows no remote script, style or connection | OK |
| 8 | SMS pumping | No SMS | Not applicable |
| 9 | Prompt injection | The prompt helper (Enhance) is a local model with no tools, files or network: what it writes only becomes prompt text, shown escaped | OK (low impact by design) |
| 10 | Weak JWT secret | No sessions or tokens of its own; your Hugging Face / Civitai keys are stored locally, never shown to the page | OK, plus #6 |
| 11 | Race condition | One job runs at a time (a queue); downloads go to a `.part` file and never overwrite | OK |
| 12 | Account pre-hijacking | No accounts | Not applicable |
| 13 | Subdomain takeover | No domain of its own; links go to GitHub, Hugging Face and Discord | Not applicable |
| 14 | SSRF | Downloads only go to fixed sites (Hugging Face, GitHub, Civitai's API); names are checked | **Hardened**: the Civitai key is only attached to Civitai's own https addresses |
| 15 | Path traversal | Every window→Mac call checks its sender and its path (pictures only, nothing is ever run); backend names are cleaned | **Fixed**: removing a prompt-helper model took an id that could hold `../` (now only known ids) |
| 16 | Password-reset poisoning | No passwords | Not applicable |
| 17 | Source maps | None shipped (the code is open source anyway) | OK |
| 18 | Clickjacking | No frames; the window can't be framed or navigated away | OK |
| 19 | Open redirect | The window can't navigate; links open in your browser only if they are https | OK |
| 20 | User enumeration | No users | Not applicable |

Also checked, because they matter more for a desktop AI app than most of the list: code hidden in model files (old
`.ckpt` pickles: loaded weights-only, or with a restricted unpickler that builds tensors only), code downloaded with a
model (`trust_remote_code`: never), HTML injected from model names, picture details or Civitai pages (everything is
shown escaped; the four places that show HTML only show the app's own text), the window's permissions (camera,
microphone and the like are refused), and the build's dependencies (pinned; adding hashes is the next step, see
`backend/requirements.lock.txt`). Tests: `backend/tests/test_security_audit.py`.
