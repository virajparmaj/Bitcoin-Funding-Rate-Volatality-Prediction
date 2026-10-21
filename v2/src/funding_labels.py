"""Checksum-verified public settlement archives; no API credentials required."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def sha256(path: Path) -> str:
    """Hash a source without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_bytes(path: Path) -> bytes:
    """Reject missing or corrupted archive/checksum pairs."""
    expected = path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return blob


def download_archives(directory: Path, start: str, end: str) -> list[Path]:
    """Cache immutable monthly archives; reject changes to existing files."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in pd.period_range(start, end, freq="M"):
        path = directory / f"BTCUSDT-fundingRate-{month}.zip"
        checksum = path.with_suffix(".zip.CHECKSUM")
        if path.exists() and checksum.exists():
            verified_bytes(path)
        else:
            url = ARCHIVE + path.name
            with urllib.request.urlopen(url, timeout=30) as response:
                blob = response.read()
            with urllib.request.urlopen(url + ".CHECKSUM", timeout=30) as response:
                check = response.read()
            if hashlib.sha256(blob).hexdigest() != check.decode().split()[0]:
                raise ValueError(f"Downloaded checksum mismatch: {url}")
            if path.exists() and path.read_bytes() != blob:
                raise ValueError(f"Refusing to replace existing archive: {path}")
            if checksum.exists() and checksum.read_bytes() != check:
                raise ValueError(f"Refusing to replace existing checksum: {checksum}")
            path.write_bytes(blob)
            checksum.write_bytes(check)
        paths.append(path)
        print(f"Verified {path.name}", flush=True)
    return paths


def load_labels(paths: list[Path], delay_minutes: float, tolerance_seconds: float):
    """Read unique scheduled labels, preserving actual calculation time and offsets.

    The narrow rounded-hour mapping reconciles millisecond archive jitter only.
    It is never applied to feature arrival times. Unexpected schedules fail closed.
    """
    frames, manifest = [], []
    for path in paths:
        with zipfile.ZipFile(io.BytesIO(verified_bytes(path))) as archive:
            names = archive.namelist()
            if len(names) != 1 or not names[0].endswith(".csv"):
                raise ValueError(f"Unexpected ZIP members: {path}")
            with archive.open(names[0]) as handle:
                frame = pd.read_csv(handle)
        if list(frame.columns) != COLUMNS or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Invalid label schema/values: {path}")
        frame["source_archive"] = path.name
        frames.append(frame)
        manifest.append(
            {
                "file": path.name,
                "url": ARCHIVE + path.name,
                "sha256": sha256(path),
                "checksum_url": ARCHIVE + path.name + ".CHECKSUM",
                "checksum_sha256": sha256(path.with_suffix(".zip.CHECKSUM")),
            }
        )
    if not frames:
        raise ValueError("No verified funding archives; run the acquire stage first")
    labels = pd.concat(frames, ignore_index=True)
    labels["calc_time"] = pd.to_datetime(labels.calc_time, unit="ms", utc=True)
    labels["event_id"] = labels.calc_time.dt.round("h")
    labels["offset_seconds"] = (labels.calc_time - labels.event_id).dt.total_seconds()
    valid = labels.offset_seconds.abs().le(tolerance_seconds)
    valid &= labels.event_id.dt.hour.isin([0, 8, 16]) & labels.funding_interval_hours.eq(8)
    if not valid.all() or labels.event_id.duplicated().any():
        raise ValueError("Ambiguous/duplicate label or unexpected funding schedule/offset")
    labels["label_available_at"] = labels.calc_time + pd.Timedelta(minutes=delay_minutes)
    labels = labels.rename(columns={"last_funding_rate": "target"})
    return labels.sort_values("event_id").reset_index(drop=True), manifest
