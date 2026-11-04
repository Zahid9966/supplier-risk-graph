"""Session fixture: a small simulated world pushed through the whole pipeline once (shared by all tests)."""
import json, os
from pathlib import Path
import pandas as pd
import pytest

from srg import config, pipeline, features, score
from srg.simulate import SimConfig, simulate


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("srg")
    old = {k: os.environ.get(k) for k in ("SRG_DATA", "SRG_REPORTS")}
    os.environ["SRG_DATA"] = str(root / "data"); os.environ["SRG_REPORTS"] = str(root / "reports")
    cfg = SimConfig(seed=11, tier_sizes=(25, 70, 140, 165), gamma=7.0)
    manifest = simulate(cfg, config.layer("raw"), config.layer("sim_truth"))
    pipeline.run_bronze(); info = pipeline.run_silver()
    from srg.cli import load_silver
    silver = load_silver(); g = features.build_features(silver)
    g.to_parquet(config.layer("gold") / "features.parquet", index=False)
    yield dict(root=root, manifest=manifest, info=info, silver=silver, g=g, cfg=cfg)
    for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
