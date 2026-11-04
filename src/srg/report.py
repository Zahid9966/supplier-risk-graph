"""Render reports/REPORT.md from evaluation + dq + leakage JSON."""
import json
from . import config

ORDER = ["base_rate", "heur_own_composite", "heur_propagated", "logit_static", "logit_own", "hgb_own", "hgb_own_struct", "logit_full", "hgb_full", "ORACLE_sim_only"]


def f(x, d=3): return "-" if x is None else f"{x:.{d}f}"


def render(ev, leak, dq_silver, manifest, sweep=None):
    r = config.reports_root(); L = []
    L.append("# Evaluation report (auto-generated)\n")
    s = ev["splits"]
    L.append(f"Task: predict a NEW supplier disruption within the next {ev['horizon_months']} months, scored monthly. "
             "All data below is **simulated** except the country covariates (see README).\n")
    L.append("| split | months | rows | positives | base rate |\n|---|---|---|---|---|")
    for k, v in s.items(): L.append(f"| {k} | {v['t_range'][0]}-{v['t_range'][1]} | {v['rows']} | {v['positives']} | {v['base_rate']:.3f} |")
    L.append("\n## Test-set results (95% supplier-clustered bootstrap CI on PR-AUC)\n")
    L.append("| model | ROC-AUC | PR-AUC [CI] | prec@5% | recall@5% | lift@5% | Brier | ECE |\n|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        m = ev["models"].get(k)
        if not m: continue
        L.append(f"| {k} | {f(m['roc_auc'])} | {f(m['pr_auc'])} [{f(m['pr_auc_ci'][0])}, {f(m['pr_auc_ci'][1])}] | {f(m['precision_at_5pct'])} | {f(m['recall_at_5pct'])} | {f(m['lift_at_5pct'],2)} | {f(m.get('brier'))} | {f(m.get('ece'))} |")
    h = ev["head_to_head_full_vs_struct"]
    for k, v in h.items():
        for dk, dv in v.items():
            if dk.startswith("pr_auc_diff"):
                L.append(f"\nPaired PR-AUC difference `{k}` {dk.replace('pr_auc_diff_vs_', 'minus ')}: mean {dv['mean']:+.4f}, 95% CI [{dv['ci'][0]:+.4f}, {dv['ci'][1]:+.4f}], positive in {dv['frac_positive']*100:.0f}% of resamples.")
    for k, v in ev["models"]["hgb_full"].items():
        if k.startswith("pr_auc_diff_vs_"):
            L.append(f"\n`hgb_full` minus `hgb_own` (graph-free): mean {v['mean']:+.4f}, 95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}].")
    L.append("\n## Ablations (remove one feature group from the main model)\n\n| variant | ROC-AUC | PR-AUC | prec@5% |\n|---|---|---|---|")
    for k, m in ev["ablations"].items(): L.append(f"| {k} | {f(m['roc_auc'])} | {f(m['pr_auc'])} | {f(m['precision_at_5pct'])} |")
    L.append("\nPermutation importance (PR-AUC drop, test, 5 shuffles): " + ", ".join(f"{k} {v:+.4f}" for k, v in ev["group_permutation_importance_pr_auc_drop"].items()))
    c = ev["calibration_main_model"]
    L.append(f"\n## Calibration (main model)\nRaw: Brier {c['raw']['brier']:.4f}, ECE {c['raw']['ece']:.4f}. Platt on validation: Brier {c['platt_on_val']['brier']:.4f}, ECE {c['platt_on_val']['ece']:.4f}. Constant base-rate Brier {c['base_rate_brier']:.4f}.")
    cs = ev["cold_start_20pct_suppliers_held_out"]
    L.append(f"\n## Cold-start suppliers (20% of suppliers never seen in training; {cs['rows']} rows, {cs['positives']} positives)\n")
    for k in ("hgb_own", "hgb_full"): L.append(f"- {k}: ROC-AUC {cs[k]['roc_auc']:.3f}, PR-AUC {cs[k]['pr_auc']:.3f}")
    L.append("\n## Leakage audit\n\n| check | result | detail |\n|---|---|---|")
    for c in leak: L.append(f"| {c['name']} | {'PASS' if c['passed'] else '**FAIL**'} | {c['detail']} |")
    L.append("\n## Data quality (silver)\n")
    bad = [x for x in dq_silver if not x["passed"]]
    L.append(f"{len(dq_silver)} checks, {len(dq_silver)-len(bad)} passed. " + (("Failing (non-critical): " + ", ".join(f"{b['name']}={b['observed']}" for b in bad)) if bad else ""))
    L.append("\n| check | severity | observed | expectation | pass |\n|---|---|---|---|---|")
    for x in dq_silver: L.append(f"| {x['name']} | {x['severity']} | {x['observed']} | {x['expectation']} | {'yes' if x['passed'] else 'NO'} |")
    L.append(f"\nInjected raw defects (simulator manifest): `{json.dumps(manifest['injected_defects'])}`")
    if sweep:
        L.append("\n## Contagion-strength sweep (does the graph help only when contagion exists?)\n")
        L.append(sweep)
    (r / "REPORT.md").write_text("\n".join(L) + "\n")
