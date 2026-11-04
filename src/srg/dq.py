"""Tiny declarative data-quality framework: named checks with severity, observed value, threshold, JSON report."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable


@dataclass
class Result:
    layer: str
    name: str
    severity: str          # critical | warn
    passed: bool
    observed: float | int | str
    expectation: str


class DQ:
    def __init__(self, layer: str):
        self.layer, self.results = layer, []

    def check(self, name: str, ok: bool, observed, expectation: str, severity: str = "critical"):
        self.results.append(Result(self.layer, name, severity, bool(ok), observed if isinstance(observed, str) else float(observed) if not isinstance(observed, int) else observed, expectation))

    def fraction_le(self, name, num, den, limit, severity="critical"):
        frac = num / den if den else 0.0
        self.check(name, frac <= limit, round(frac, 4), f"<= {limit}", severity)

    @property
    def failed_critical(self):
        return [r for r in self.results if not r.passed and r.severity == "critical"]

    def raise_if_failed(self):
        if self.failed_critical:
            raise RuntimeError("critical data-quality failures: " + "; ".join(f"{r.name}={r.observed} (want {r.expectation})" for r in self.failed_critical))

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self.results], indent=2))
