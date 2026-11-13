"""Archive download, extraction, and installation helpers for dataset loaders."""

import contextlib
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

RETRY_DELAY = 10.0


def fetch_archive(
    url: str,
    root: Path,
    client: httpx.Client | None = None,
    *,
    require_length: bool = False,
    retries: int = 0,
) -> Path:
    """Stream ``url`` to ``root/.downloads/<name>`` through a ``.part`` file.

    An archive completed by an earlier call is returned without a request. An existing
    ``.part`` file is resumed with a ``Range`` request: a 206 response is appended, a 200
    response restarts the file, and a 416 response discards it and restarts. An unencoded
    body whose size differs from ``Content-Length`` is discarded with ``ValueError``;
    ``require_length`` also rejects responses without that header. ``retries`` repeats the
    request, resuming, after an ``httpx.TransportError`` such as a dropped connection,
    waiting ``RETRY_DELAY`` seconds first.
    """
    target = root / ".downloads" / Path(urlparse(url).path).name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with contextlib.ExitStack() as stack:
        if client is None:
            client = stack.enter_context(httpx.Client(follow_redirects=True, timeout=60))
        for attempt in range(retries + 1):
            try:
                _stream(client, url, partial, require_length)
                break
            except httpx.TransportError:
                if attempt == retries:
                    raise
                time.sleep(RETRY_DELAY)
    partial.replace(target)
    return target


def _stream(client: httpx.Client, url: str, partial: Path, require_length: bool) -> None:
    """Fetch ``url`` into ``partial``, resuming from its current size."""
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code != 416:
            _write(response, url, partial, offset, require_length)
            return
    # The .part file is complete or stale for this range; restart without Range.
    partial.unlink()
    _stream(client, url, partial, require_length)


def _write(
    response: httpx.Response, url: str, partial: Path, offset: int, require_length: bool
) -> None:
    response.raise_for_status()
    resumed = response.status_code == 206
    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
        partial.unlink()
        raise ValueError(f"{url}: Content-Range does not start at byte {offset}")
    expected = response.headers.get("Content-Length")
    if expected is None and require_length:
        raise ValueError(f"{url} sent no Content-Length")
    received = 0
    with partial.open("ab" if resumed else "wb") as stream:
        for chunk in response.iter_bytes():
            received += stream.write(chunk)
    # Content-Length counts encoded bytes, so only identity bodies are comparable.
    encoded = response.headers.get("Content-Encoding", "identity") != "identity"
    if expected is not None and not encoded and received != int(expected):
        partial.unlink()
        raise ValueError(f"{url}: received {received} bytes, Content-Length {expected}")


def extract(archive: Path, destination: Path) -> None:
    """Unpack a tar (``data`` filter) or zip archive into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")


def install_archive(
    archive: Path, root: Path, inner: str, last: str, nested: str | None = None
) -> None:
    """Extract ``archive`` to a staging directory and move its ``inner`` entries into ``root``.

    A ``nested`` archive inside ``inner`` is unpacked in place and deleted before moving.
    The ``last`` entry moves after all others, so its presence marks a complete install.
    Existing entries of the same name are replaced; the staging directory and archive
    are deleted afterwards.
    """
    staging = archive.with_name(archive.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    extract(archive, staging)
    if nested is not None:
        extract(staging / inner / nested, staging / inner)
        (staging / inner / nested).unlink()
    for entry in sorted((staging / inner).iterdir(), key=lambda p: p.name == last):
        target = root / entry.name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        entry.replace(target)
    shutil.rmtree(staging)
    archive.unlink()
