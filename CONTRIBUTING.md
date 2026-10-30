# Contributing

Thanks for helping! Bug reports, fixes and ideas are all welcome.

## Set up

Requirements: a Mac with Apple Silicon, macOS 13 or newer, Python 3.10+ (3.12 recommended), Node.js 18+.

```
./setup.sh      # Python environment (exact versions from backend/requirements.lock.txt) + Electron
./start.sh      # run the app; it reloads when you edit files in app/renderer or backend/
```

## Tests

```
cd backend && .venv/bin/python -m unittest discover -s tests -t .
```

`Self-test training.command` (or `backend/.venv/bin/python backend/tools/selftest_train.py`) runs a short real LoRA
merge and embedding training on your Mac, and `Enhance Speed Test.command` times the enhancer; both need models
downloaded and are for checking changes to those features.

Please add or update tests for backend changes, and keep the tests passing.

## Guidelines

- Keep the app local and private: no telemetry, no new network calls without a clear reason and a Strict offline check.
- New dependencies: add them to `backend/requirements.txt`, then regenerate `backend/requirements.lock.txt` from a
  working environment, and say why in the pull request.
- Vendored code must keep its licence header and be listed in `THIRD_PARTY_NOTICES.md`; only licences compatible
  with Apache-2.0 (MIT, BSD, Apache-2.0).
- Never commit model weights, pictures, your library database, logs, API keys or tokens (`.gitignore` covers the usual
  files). Run a secret scanner such as `gitleaks dir .` before pushing.
- Security problems: see `SECURITY.md` (please report privately).

## Building the app

`Build Protoart Studio.command` (or `bash packaging/build_mac_app.sh`) builds `dist/Protoart Studio.app`;
`Make DMG for friends.command` builds a DMG plus its `.sha256` checksum. `Rebuild installer.command` runs the
self-tests first (`packaging/run_tests.py`: plain `test_` functions and unittest classes), then builds the DMG.
