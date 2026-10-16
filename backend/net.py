"""Privacy: Protoart Studio works offline.

* Telemetry in the Hugging Face libraries is switched off.
* The model libraries run in offline mode, so loading a model you already have never contacts the
  internet (before, every load asked Hugging Face whether there was a newer version — slow on VPNs).
* Only when a job needs a file that isn't on this Mac yet (a model's first download) is the
  connection opened, for that job only, and only to download from Hugging Face.
* "Strict offline" (Settings) blocks even that: nothing ever leaves the Mac.
Must be imported before torch / diffusers / transformers / huggingface_hub.
"""
import os

for _k, _v in {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    # no token is sent unless you saved one in Protoart Studio (Settings → Models → Hugging Face)
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0" if os.environ.get("PROTOART_OWN_HF_TOKEN") == "1" else "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "GRADIO_ANALYTICS_ENABLED": "False", "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ[_k] = _v
# Downloads must never hang: give up on a connection that stays silent (the download resumes next time).
# Plain HTTP downloads (resumable, one connection per file) instead of the hf_xet downloader, which opens
# dozens of connections at once and stalled behind per-app firewalls and slow Wi-Fi.
for _k, _v in {"HF_HUB_DOWNLOAD_TIMEOUT": "60", "HF_HUB_ETAG_TIMEOUT": "30", "HF_HUB_DISABLE_XET": "1"}.items():
    os.environ.setdefault(_k, _v)

strict = False      # Settings -> Privacy -> Strict offline
_online = False

OFFLINE_HINTS = ("offline", "local_files_only", "localentrynotfound", "cannot find the requested files",
                 "couldn't connect", "could not connect", "not found in the cache", "outgoing traffic has been disabled",
                 "we couldn't connect", "connection error", "not cached",
                 # diffusers' own wording when a file (e.g. an image prompt adapter) isn't downloaded yet
                 "does not appear to have a file named")


class NeedsDownload(Exception):
    pass


def is_online():
    return _online


def set_online(on):
    """Open (on=True) or close the connection for the model libraries."""
    global _online
    _online = bool(on) and not strict
    if _online:
        try:
            ensure_downloads_work()
        except Exception as e:
            print("[net] connection check failed:", e, flush=True)
    val = "0" if _online else "1"
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = val
    import sys
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        hub.HF_HUB_OFFLINE = not _online
    dc = sys.modules.get("diffusers.utils.constants")
    if dc is not None and hasattr(dc, "HF_HUB_OFFLINE"):
        dc.HF_HUB_OFFLINE = not _online
    for name in ("transformers.utils.hub", "transformers.utils.import_utils"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "_is_offline_mode"):
            try:
                m._is_offline_mode = not _online
            except Exception:
                pass


def looks_like_missing_download(exc):
    """True if an error means 'a file isn't on this Mac and we weren't allowed to download it'."""
    seen = 0
    e = exc
    while e is not None and seen < 6:
        name = type(e).__name__.lower()
        msg = str(e).lower()
        if "offlinemode" in name or "localentrynotfound" in name or any(h in msg for h in OFFLINE_HINTS):
            return True
        e = e.__cause__ or e.__context__
        seen += 1
    return False


# ---------------------------------------------------------------------------------------------------------
# Firewall fallback: download through macOS's own curl.
#
# Per-app firewalls (Little Snitch, LuLu, Radio Silence…) know curl but not the Python inside Protoart
# Studio.app, and quietly block it ("[Errno 9] Bad file descriptor"). When that happens, Hugging Face
# downloads and other one-time downloads are routed through /usr/bin/curl instead. Same files, same
# servers; only the program that opens the connection changes.
CURL = "/usr/bin/curl"
_curl_mode = None   # None = not checked yet, False = Python connects fine, True = use curl
_urllib_curl = False  # urllib downloads already routed through curl


def python_can_connect(host="huggingface.co", port=443, timeout=6):
    import socket
    try:
        socket.create_connection((host, port), timeout).close()
        return True
    except OSError as e:
        print(f"[net] Python can't open connections ({e}); downloads will go through curl", flush=True)
        return False


def _curl_open(method, url, headers=None, body=None, follow=False):
    """Run curl and return (status, reason, [(name, value)], proc) once the final response headers have
    arrived; the body can then be read from proc.stdout."""
    import subprocess
    # --speed-limit/--speed-time: give up if fewer than 1 KB/s arrive for 2 minutes (a stalled connection),
    # so a download can't hang forever; Hugging Face downloads resume where they stopped next time.
    cmd = [CURL, "-sS", "-i", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
           "-X", method, url]
    if method == "HEAD":
        cmd = [CURL, "-sS", "-i", "--connect-timeout", "30", "--max-time", "60", "--head", url]
    if follow:
        cmd.insert(1, "-L")
    for k, v in (headers or []):
        if k.lower() in ("host", "content-length", "accept-encoding", "connection", "transfer-encoding"):
            continue
        cmd += ["-H", f"{k}: {v}"]
    if body:
        cmd += ["--data-binary", "@-"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if body else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if body:
        proc.stdin.write(body)
        proc.stdin.close()
    out = proc.stdout
    while True:  # header blocks: skip "100 Continue" and (with -L) redirects, keep the last one
        line = out.readline()
        if not line:
            err = proc.stderr.read().decode("utf-8", "replace").strip()
            proc.wait()
            raise ConnectionError(f"curl couldn't connect: {err or 'no response'}")
        parts = line.decode("latin-1").strip().split(" ", 2)
        status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        reason = parts[2] if len(parts) > 2 else ""
        hdrs = []
        while True:
            h = out.readline()
            if not h or h in (b"\r\n", b"\n"):
                break
            k, _, v = h.decode("latin-1").partition(":")
            hdrs.append((k.strip(), v.strip()))
        if status == 100 or (follow and 300 <= status < 400 and any(k.lower() == "location" for k, _ in hdrs)):
            continue
        if method == "HEAD":
            proc.wait()
        return status, reason, hdrs, proc


def _install_curl_hf():
    """huggingface_hub (and so transformers / diffusers) downloads through curl."""
    import httpx2
    from huggingface_hub.utils import _http as hf_http

    class _Body(httpx2.SyncByteStream):
        def __init__(self, proc):
            self.proc = proc

        def __iter__(self):
            while True:
                chunk = self.proc.stdout.read(1 << 20)
                if not chunk:
                    break
                yield chunk

        def close(self):
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass

    class CurlTransport(httpx2.BaseTransport):
        def handle_request(self, request):
            body = request.read() if request.method not in ("GET", "HEAD") else None
            try:
                status, _, hdrs, proc = _curl_open(request.method, str(request.url), list(request.headers.items()), body)
            except ConnectionError as e:
                raise httpx2.ConnectError(str(e), request=request)
            # curl already removed any transfer encoding
            hdrs = [(k, v) for k, v in hdrs if k.lower() != "transfer-encoding"]
            return httpx2.Response(status, headers=hdrs, stream=_Body(proc), request=request)

    def factory():
        return httpx2.Client(transport=CurlTransport(), event_hooks={"request": [hf_http.hf_request_event_hook]},
                             follow_redirects=True, timeout=None)

    hf_http.set_client_factory(factory)
    # large files normally use the hf_xet downloader, which opens its own connections: use plain HTTP instead
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import sys
    c = sys.modules.get("huggingface_hub.constants")
    if c is not None and hasattr(c, "HF_HUB_DISABLE_XET"):
        c.HF_HUB_DISABLE_XET = True


def _install_curl_urllib():
    """Plain urllib downloads (face-restore weights, face detector) through curl as well."""
    import io
    import urllib.request
    from email.message import Message
    from urllib.response import addinfourl

    class CurlHandler(urllib.request.BaseHandler):
        handler_order = 100  # before the normal HTTP(S) handlers

        def _open(self, req):
            hdrs = list(req.header_items())
            status, reason, h, proc = _curl_open(req.get_method(), req.full_url, hdrs, req.data, follow=True)
            msg = Message()
            for k, v in h:
                msg[k] = v
            if status >= 400:
                data = proc.stdout.read()
                proc.wait()
                raise urllib.error.HTTPError(req.full_url, status, reason, msg, io.BytesIO(data))
            resp = addinfourl(proc.stdout, msg, req.full_url, status)
            resp.msg = reason
            return resp

        def https_open(self, req):
            return self._open(req)

        def http_open(self, req):
            return self._open(req)

    import urllib.error  # noqa: F401
    urllib.request.install_opener(urllib.request.build_opener(CurlHandler()))


def use_curl_for_urllib():
    """Route plain urllib downloads through curl from now on (a firewall blocked Python's own connection to a
    site even though the first check, to huggingface.co, got through). Returns True if curl is available."""
    global _urllib_curl
    if not os.path.exists(CURL):
        return False
    if _urllib_curl:
        return False  # already going through curl: nothing new to try
    try:
        _install_curl_urllib()
    except Exception as e:
        print("[net] couldn't route downloads through curl:", e, flush=True)
        return False
    _urllib_curl = True
    print("[net] a site was blocked for Python; urllib downloads now go through curl", flush=True)
    return True


def is_blocked_connection(exc):
    """True for the errors a per-app firewall or VPN gives Python ('[Errno 9] Bad file descriptor',
    connection refused/reset before any answer), as opposed to an answer from the server."""
    import errno
    import urllib.error
    if isinstance(exc, urllib.error.HTTPError):
        return False
    e = getattr(exc, "reason", exc)
    return isinstance(e, OSError) and getattr(e, "errno", None) in (
        errno.EBADF, errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED, errno.EPERM, errno.EACCES, errno.ENETUNREACH,
        errno.EHOSTUNREACH) or any(t in str(e).lower() for t in ("bad file descriptor", "connection abort"))


def ensure_downloads_work():
    """Called when going online: check once whether Python may connect; if not, switch to curl."""
    global _curl_mode
    if _curl_mode is not None:
        return _curl_mode
    if os.environ.get("PROTOART_FORCE_CURL") == "1" or not python_can_connect():
        if os.path.exists(CURL):
            try:
                _install_curl_hf()
            except Exception as e:
                print("[net] couldn't route Hugging Face downloads through curl:", e, flush=True)
            use_curl_for_urllib()
            _curl_mode = True
        else:
            _curl_mode = False
    else:
        _curl_mode = False
    return _curl_mode


# ------------------------------------------------------------------ Hugging Face sign-in (gated models)
HF_TOKEN_RE = r"hf_[A-Za-z0-9]{20,80}"


def apply_hf_token(token):
    """Use (or stop using) your Hugging Face access token for downloads. Kept in this process's environment only
    (not written to ~/.cache/huggingface); huggingface_hub sends it to huggingface.co, never to the file servers
    it redirects to."""
    import os
    import sys
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        os.environ.pop("HF_TOKEN", None)
    # Normally no token is ever sent (HF_HUB_DISABLE_IMPLICIT_TOKEN, so one saved by other apps stays private);
    # with your own token saved here, it is sent to huggingface.co (the env token wins over any saved elsewhere).
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0" if token else "1"
    os.environ["PROTOART_OWN_HF_TOKEN"] = "1" if token else "0"   # (helper processes started later follow this)
    consts = sys.modules.get("huggingface_hub.constants")
    if consts is not None:
        consts.HF_HUB_DISABLE_IMPLICIT_TOKEN = not token


def hf_auth_problem(err, repo=None):
    """A friendly message when Hugging Face refused a download because the model is gated or the token is bad;
    None for any other error."""
    import os
    name = type(err).__name__
    msg = str(err)
    gated = name == "GatedRepoError" or "gated" in msg.lower() or "awaiting a review" in msg.lower()
    auth = gated or name in ("HfHubHTTPError", "RepositoryNotFoundError", "HTTPError") and (
        "401" in msg or "403" in msg) or " 401 " in f" {msg} " or "Unauthorized" in msg
    if not auth:
        return None
    what = repo or "This model"
    if os.environ.get("HF_TOKEN"):
        return (f"Hugging Face turned the download of {what} down. Open the model's page on huggingface.co while "
                "signed in and accept its licence (gated models need this once; some are reviewed by their owner "
                "first). If you already did, your access token may be wrong or deleted: make a new \"Read\" token "
                "(huggingface.co → Settings → Access Tokens) and paste it in Settings → Models → Hugging Face.")
    return (f"{what} is gated on Hugging Face: it needs a free huggingface.co account. Sign in there, open the "
            "model's page and accept its licence, then make a \"Read\" access token (huggingface.co → Settings → "
            "Access Tokens) and paste it in Settings → Models → Hugging Face. Then try again.")
