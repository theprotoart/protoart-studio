"""⚡ Models kept on this Mac: copies, on the Mac's own drive, of downloaded models whose files are on an external drive.

Models downloaded to an external drive (Settings → where models download) load from it every time, and a slow
drive makes that a minute or more (SD 1.5 measured 63-116 s from the drive vs 3-7 s from the Mac). A kept copy is
the model's downloaded folder copied into ~/.protoartstudio/fast-copies/<models--org--name>--<revision>; the app
loads from it instead. The drive's files are only read; removing a kept copy deletes only this app-made copy.
The fast engine names its copies the same way (mlx_worker._internal_copy), so both see the same ones."""
import os
import shutil
from pathlib import Path

import paths

DONE = ".protoart-done"


def root():
    return Path(os.environ.get("PROTOART_HOME") or paths.HOME) / "fast-copies"


def snapshot(repo, revision=None):
    """The downloaded folder of `repo` (in the Hugging Face cache), or None if it isn't downloaded."""
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except Exception:  # noqa: BLE001  (not downloaded, or only partly)
        return None


def dest_for(snap):
    return root() / f"{snap.parent.parent.name}--{snap.name}"


def kept_path(repo):
    """The kept copy of `repo` (any revision), or None."""
    d = root()
    if not d.is_dir():
        return None
    prefix = "models--" + repo.replace("/", "--") + "--"
    for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.name.startswith(prefix) and (p / DONE).exists():
            return p
    return None


def on_external(snap):
    try:
        return str(Path(snap).resolve()).startswith("/Volumes/") or any(
            str(f.resolve()).startswith("/Volumes/") for f in list(Path(snap).rglob("*"))[:50] if f.is_file())
    except OSError:
        return False


def size(snap):
    total = 0
    for f in Path(snap).rglob("*"):
        try:
            if f.is_file():
                total += f.resolve().stat().st_size
        except OSError:
            pass
    return total


def keep(repo, revision=None, progress=None, cancelled=None):
    """Copy `repo`'s downloaded folder onto this Mac. Returns the copy's folder."""
    snap = snapshot(repo, revision)
    if snap is None:
        raise ValueError("It isn't downloaded yet: make a picture with it first.")
    dest = dest_for(snap)
    if (dest / DONE).exists():
        return dest
    files = [f for f in snap.rglob("*") if f.is_file()]
    need = sum(f.resolve().stat().st_size for f in files)
    dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest.parent).free
    if free < need + (5 << 30):
        raise ValueError(f"Not enough space on this Mac: it needs {need / 1e9:.1f} GB plus 5 GB spare, and "
                         f"{free / 1e9:.1f} GB is free.")
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    done = 0
    try:
        for f in files:
            if cancelled and cancelled():
                raise InterruptedError
            out = part / f.relative_to(snap)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f.resolve(), out)
            done += out.stat().st_size
            if progress:
                progress(done, need)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    (part / DONE).write_text(str(snap.resolve()))
    shutil.rmtree(dest, ignore_errors=True)
    part.rename(dest)
    return dest


def remove(repo):
    """Delete this app's copy of `repo` on this Mac (the drive's files stay). Returns how many GB were freed."""
    freed = 0
    d = root()
    if not d.is_dir():
        return 0
    prefix = "models--" + repo.replace("/", "--") + "--"
    for p in d.iterdir():
        if p.name.startswith(prefix) and (p / DONE).exists():  # only folders this app made
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p, ignore_errors=True)
    return round(freed / 1e9, 1)
