# Releasing Protoart Studio

Releases are built on an Apple Silicon Mac and published as GitHub Releases. The repository holds source code only:
**never commit the `.app`, the `.dmg`, model weights, pictures or any key / token.**

## 1. Prepare

1. Update the version in `app/package.json` (`"version"`), and add a section for it at the top of `CHANGELOG.md`.
2. Make sure the working tree is clean and up to date: `git status`, `git pull`.
3. Run the tests: `cd backend && .venv/bin/python -m unittest discover -s tests -t .`
4. Scan for secrets before every push: `gitleaks dir .` (or `gitleaks git .`). It must report no leaks
   (the one fake test key is listed in `.gitleaksignore`).

## 2. Build

```bash
bash packaging/make_dmg.sh        # or double-click "Make DMG for friends.command"
```

This rebuilds `dist/Protoart Studio.app` (with its own Python, ~1.8 GB), checks that it doesn't depend on anything only
your Mac has, signs it ad hoc, and writes:

- `dist/Protoart-Studio-<version>.dmg` (~530 MB)
- `dist/Protoart-Studio-<version>.dmg.sha256`

It needs about 4 GB free while it works (it checks first). Open the DMG once and check the install window and that the app starts.

## 3. Publish (only after review)

```bash
git tag -a v<version> -m "Protoart Studio <version>"
git push origin main --tags
gh release create v<version> \
  "dist/Protoart-Studio-<version>.dmg" "dist/Protoart-Studio-<version>.dmg.sha256" \
  --title "Protoart Studio <version>" --notes-file <notes.md>
```

GitHub limits release files to 2 GB each, so the ~530 MB DMG fits. Paste the SHA-256 line into the release notes too.

Release notes should say:

- Apple Silicon Macs only (M1 or newer), macOS 13+.
- Test builds are not notarised: first launch needs **System Settings → Privacy & Security → Open Anyway**.
- No model weights are included; models download on first use and each has its own licence.
- How to check the download: `shasum -a 256 Protoart-Studio-<version>.dmg`.

## Signing and notarisation (optional, later)

With an Apple Developer account ($99/year) the app can be signed with a Developer ID and notarised, so macOS opens it
without the Open Anyway step. That needs changes to `packaging/build_mac_app.sh` (codesign with the Developer ID and
hardened runtime, then `xcrun notarytool submit` and `xcrun stapler staple` on the DMG). Keep the signing certificate
and the notarisation app-specific password out of the repository (Keychain or CI secrets only).
