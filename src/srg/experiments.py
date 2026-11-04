"""Contagion-strength sweep: does the graph help only when the world actually has contagion?

For each (gamma, seed): simulate -> bronze -> silver -> gold -> train {hgb_own, hgb_own_struct, hgb_full} -> test PR-AUC.
gamma=0 is the NO-CONTAGION CONTROL: graph-contagion features should add nothing there.
"""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config, pipeline, features, evaluate, models as M
from .simulate import SimConfig, simulate


def one(gamma, seed, root: Path):
    old = os.environ.get("SRG_DATA"), os.environ.get("SRG_REPORTS")
    os.environ["SRG_DATA"] = str(root); os.environ["SRG_REPORTS"] = str(root / "rep")
    try:
        man = simulate(SimConfig(seed=seed, gamma=gamma), config.layer("raw"), config.layer("sim_truth"))
        pipeline.run_bronze(); pipeline.run_silver()
        from .cli import load_silver
        g = features.build_features(load_silver())
        sp = evaluate.split(g); tr, va, te = sp["train"], sp["val"], sp["test"]
        out = dict(gamma=gamma, seed=seed, contagion_share=man["contagion_share_of_starts"], test_pos=int(te.y.sum()))
        for name in ("hgb_own", "hgb_own_struct", "hgb_full"):
            spec = M.SPECS[name]; m, _ = evaluate.fit_select(spec, tr, va); p = m.predict_proba1(te[spec[0]])
            out[name + "_pr"] = float(average_precision_score(te.y, p)); out[name + "_auc"] = float(roc_auc_score(te.y, p))
        return out
    finally:
        for k, v in zip(("SRG_DATA", "SRG_REPORTS"), old):
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def sweep(gammas, seeds):
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for gm in gammas:
            for sd in seeds:
                r = one(gm, sd, Path(td) / f"g{gm}_s{sd}"); rows.append(r); print(r, flush=True)
    d = pd.DataFrame(rows); d["pr_gain_full_vs_own"] = d.hgb_full_pr - d.hgb_own_pr
    d["pr_gain_full_vs_struct"] = d.hgb_full_pr - d.hgb_own_struct_pr
    agg = d.groupby("gamma").agg(contagion_share=("contagion_share", "mean"), own_pr=("hgb_own_pr", "mean"), struct_pr=("hgb_own_struct_pr", "mean"),
                                 full_pr=("hgb_full_pr", "mean"), gain_vs_own=("pr_gain_full_vs_own", "mean"), gain_vs_own_sd=("pr_gain_full_vs_own", "std"),
                                 gain_vs_struct=("pr_gain_full_vs_struct", "mean"), gain_vs_struct_sd=("pr_gain_full_vs_struct", "std"),
                                 seeds_full_beats_struct=("pr_gain_full_vs_struct", lambda x: int((x > 0).sum()))).reset_index()
    md = ["Mean test PR-AUC over seeds " + str(sorted(set(d.seed))) + " (each seed = a new simulated world; gamma = contagion strength, 0 = control).\n",
          "| gamma | contagion share of starts | hgb_own | hgb_own_struct | hgb_full | full - own (sd) | full - struct (sd) | seeds where full > struct |", "|---|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        md.append(f"| {r.gamma:g} | {r.contagion_share:.2f} | {r.own_pr:.3f} | {r.struct_pr:.3f} | {r.full_pr:.3f} | {r.gain_vs_own:+.4f} ({r.gain_vs_own_sd:.4f}) | {r.gain_vs_struct:+.4f} ({r.gain_vs_struct_sd:.4f}) | {r.seeds_full_beats_struct}/{len(seeds)} |")
    rep = config.reports_root(); rep.mkdir(parents=True, exist_ok=True)
    (rep / "sweep.md").write_text("\n".join(md) + "\n"); d.to_csv(rep / "sweep.csv", index=False)
    print("\n".join(md))
