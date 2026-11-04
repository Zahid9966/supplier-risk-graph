"""Command line: python -m srg.cli <simulate|bronze|silver|gold|evaluate|score|run|sweep|serve>"""
from __future__ import annotations
import argparse, json, shutil
import pandas as pd

from . import config, pipeline, features, evaluate, leakage, score, report
from .simulate import SimConfig, simulate

SILVER = dict(suppliers="suppliers", edges="edges", signals="signals", incidents="incidents", bom="bom", country_indicators="country_indicators")


def load_silver():
    s = config.layer("silver")
    return {k: pd.read_parquet(s / f"{v}.parquet") for k, v in SILVER.items()}


def load_oracle():
    p = config.data_root() / "sim_truth" / "dynamics.parquet"
    if not p.exists(): return None
    d = pd.read_parquet(p).rename(columns={"month": "t"})
    u = pd.read_parquet(config.data_root() / "sim_truth" / "supplier_latents.parquet")
    d = d.merge(u[["supplier_id", "true_u"]], on="supplier_id")
    return d.rename(columns={"p_upstream": "true_p_upstream", "active": "true_active", "fin_stress": "true_fin_stress"})[
        ["t", "supplier_id", "true_p_upstream", "true_active", "true_fin_stress", "true_u"]].astype({"true_active": float})


def cmd_simulate(a):
    m = simulate(SimConfig(seed=a.seed, gamma=a.gamma), config.layer("raw"), config.layer("sim_truth"))
    print(f"simulated {m['n_suppliers']} suppliers, {m['n_edges']} edges, {m['n_incidents']} incidents; contagion share of starts {m['contagion_share_of_starts']:.2f}")


def cmd_gold(a=None):
    g = features.build_features(load_silver()); g.to_parquet(config.layer("gold") / "features.parquet", index=False)
    print(f"gold features {g.shape}, labelled prevalence {g.y.mean():.4f}")


def cmd_evaluate(a=None, sweep_md=None):
    silver, g = load_silver(), pd.read_parquet(config.layer("gold") / "features.parquet")
    ev = evaluate.run_evaluation(g, oracle=load_oracle())
    lk = leakage.run_all(silver, g, ev)
    rep = config.reports_root(); rep.mkdir(parents=True, exist_ok=True)
    (rep / "metrics.json").write_text(json.dumps(ev, indent=2)); (rep / "leakage.json").write_text(json.dumps(lk, indent=2))
    dq = json.loads((rep / "dq_silver.json").read_text())
    man = json.loads((config.data_root() / "sim_truth" / "manifest.json").read_text())
    sweep = (rep / "sweep.md").read_text() if (rep / "sweep.md").exists() else None
    report.render(ev, lk, dq, man, sweep)
    for c in lk: print(("PASS " if c["passed"] else "FAIL ") + c["name"] + ": " + c["detail"])
    m = ev["models"]
    print({k: (round(m[k]["roc_auc"], 3), round(m[k]["pr_auc"], 3)) for k in m})
    return ev, lk


def cmd_report(a=None):
    rep = config.reports_root()
    sweep = (rep / "sweep.md").read_text() if (rep / "sweep.md").exists() else None
    report.render(json.loads((rep / "metrics.json").read_text()), json.loads((rep / "leakage.json").read_text()),
                  json.loads((rep / "dq_silver.json").read_text()),
                  json.loads((config.data_root() / "sim_truth" / "manifest.json").read_text()), sweep)
    print("wrote", rep / "REPORT.md")


def cmd_score(a=None):
    silver, g = load_silver(), pd.read_parquet(config.layer("gold") / "features.parquet")
    mp = config.reports_root() / "metrics.json"
    info = score.write_serving_artifacts(silver, g, json.loads(mp.read_text()) if mp.exists() else None)
    print("scored", info)


def cmd_run(a):
    if a.fresh:
        for d in ("raw", "bronze", "silver", "gold", "sim_truth"): shutil.rmtree(config.data_root() / d, ignore_errors=True)
    cmd_simulate(a); print("bronze", pipeline.run_bronze()); print("silver", pipeline.run_silver()); cmd_gold(); cmd_evaluate(); cmd_score()


def cmd_sweep(a):
    from .experiments import sweep
    sweep(a.gammas, a.seeds)


def cmd_serve(a):
    import uvicorn
    uvicorn.run("srg.api:app", host=a.host, port=a.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="srg"); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate"); s.add_argument("--seed", type=int, default=42); s.add_argument("--gamma", type=float, default=SimConfig.gamma); s.set_defaults(f=cmd_simulate)
    sub.add_parser("bronze").set_defaults(f=lambda a: print(pipeline.run_bronze()))
    sub.add_parser("silver").set_defaults(f=lambda a: print(pipeline.run_silver()))
    sub.add_parser("gold").set_defaults(f=cmd_gold)
    sub.add_parser("evaluate").set_defaults(f=cmd_evaluate)
    sub.add_parser("score").set_defaults(f=cmd_score)
    sub.add_parser("report").set_defaults(f=cmd_report)
    s = sub.add_parser("run"); s.add_argument("--seed", type=int, default=42); s.add_argument("--gamma", type=float, default=SimConfig.gamma); s.add_argument("--fresh", action="store_true"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("sweep"); s.add_argument("--gammas", type=float, nargs="+", default=[0, 3.5, 7, 10]); s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); s.set_defaults(f=cmd_sweep)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.set_defaults(f=cmd_serve)
    a = p.parse_args(argv); a.f(a)


if __name__ == "__main__":
    main()
