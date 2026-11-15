"""Download only the pinned H3 components; resume interrupted transfers."""
import fcntl
import shutil
from pathlib import Path
import httpx

from .models import manifest_for, sha256


def _download_file(client, url, dest, entry):
    partial = dest.with_name(dest.name + ".part")
    size = partial.stat().st_size if partial.exists() else 0
    if size > entry["bytes"]:
        raise ValueError(f"Oversized partial file: {partial}")
    if shutil.disk_usage(dest.parent).free < entry["bytes"] - size:
        raise OSError(f"Insufficient free disk space for {dest.name}")
    if size < entry["bytes"]:
        headers = {"Range": f"bytes={size}-"} if size else {}
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            if response.status_code == 206:
                if not response.headers.get("content-range", "").startswith(f"bytes {size}-"):
                    raise ValueError(f"Server returned an invalid byte range for {dest.name}")
            else:
                size = 0
            with partial.open("ab" if size else "wb") as out:
                for chunk in response.iter_bytes(1024**2):
                    size += len(chunk)
                    if size > entry["bytes"]:
                        raise ValueError(f"Response exceeded expected size for {dest.name}")
                    out.write(chunk)
    if partial.stat().st_size != entry["bytes"]:
        raise OSError(f"Incomplete download of {dest.name}. Run fetch again to resume.")
    if sha256(partial) != entry["sha256"]:
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch downloading {dest.name}")
    partial.replace(dest)


def fetch(directory: Path, source="modelscope", *, precision="int8"):
    return fetch_manifest(directory, manifest_for(precision), source)


def fetch_manifest(directory: Path, manifest, source="modelscope"):
    """Shared pinned-asset transport for model and adapter manifests."""
    if source not in ("modelscope", "huggingface"):
        raise ValueError(f"Unknown model source: {source}")
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".h3-fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing_bytes = 0
        for entry in manifest["files"]:
            dest = directory / entry["path"]
            if not dest.exists():
                partial = dest.with_name(dest.name + ".part")
                missing_bytes += max(0, entry["bytes"] - (partial.stat().st_size if partial.exists() else 0))
        if missing_bytes > shutil.disk_usage(directory).free:
            raise OSError(f"Need another {missing_bytes:,} available bytes to fetch this model")
        with httpx.Client(follow_redirects=True, timeout=90) as client:
            for entry in manifest["files"]:
                dest = directory / entry["path"]
                if dest.exists():
                    if dest.stat().st_size != entry["bytes"] or sha256(dest) != entry["sha256"]:
                        raise ValueError(f"Existing file differs from pinned weights: {dest}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                _download_file(client, entry[source], dest, entry)
    return directory
