"""Paths and global constants. Everything is relative to a data root so tests/experiments can redirect it."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "external"          # committed, real public snapshots
HORIZON = 3                                     # label: disruption starts within next H months
WARMUP = 6                                      # months of history needed before first scored month


def data_root() -> Path:
    return Path(os.environ.get("SRG_DATA", ROOT / "data"))


def reports_root() -> Path:
    return Path(os.environ.get("SRG_REPORTS", ROOT / "reports"))


def layer(name: str) -> Path:
    p = data_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p
