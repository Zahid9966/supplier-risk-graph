# supplier-risk-graph

Multi-tier supplier **disruption early-warning**: a bronze → silver → gold data pipeline with data-quality gates, a supplier dependency **graph with risk propagation**, an evaluation harness built around leakage prevention, a FastAPI service and a small dashboard.

> **Read this first: the supplier network is simulated.** There is no public dataset of who supplies whom or when suppliers failed. The *only* real data here is country-level (World Bank, OFAC). Every accuracy number below is a statement about *this simulator*, not about real supply chains. What the project demonstrates is the engineering and the method — including an experiment that shows when graph features help and when they provably don't.

## What is real, what is synthetic

| Piece | Status | Detail |
|---|---|---|
| Country governance (political stability, government effectiveness, rule of law) | **Real** | World Bank WGI, latest year ≤ 2024, API snapshot in `data/external/` |
| Country logistics quality | **Real** | World Bank Logistics Performance Index (overall) |
| Sanctions exposure by country | **Real** (crude proxy) | OFAC SDN address file: number of sanctioned-party addresses per country, 25,166 rows |
| Suppliers, names, sectors, sizes | Synthetic | 2,000 suppliers, 4 tiers (100/300/700/900) |
| Supply links, dependency shares, alternative sources | Synthetic | 3,236 links, preferential attachment (hubs), 8% skip-tier links, 12% links appear mid-history |
| Monthly signals (on-time delivery, financial score, negative news, price volatility) | Synthetic | noisy, 8–40% missing by tier, quarterly-stale financials, 2–40 day ingestion lag, late corrections |
| Disruption incidents | Synthetic | 1,303 over 60 months, from the generative process below |
| Product BOM / revenue | Synthetic | 16 products |

`python scripts/fetch_external.py` re-downloads the real snapshot (provenance and retrieval time in `data/external/SOURCES.json`). The pipeline uses the committed snapshot so a clean clone is reproducible offline. I did not use UN Comtrade or GDELT: neither provides supplier-level links or incident labels, so they could not replace the simulator, and bolting them on as decoration would blur the real/synthetic line.

## The simulator (why it is not trivially separable)

Monthly hazard of a *new* disruption for supplier *i*:

```
logit h_i(t) = b0 + 0.55·country_risk_i + 0.5·sector_i + 0.6·u_i          static; u_i is UNOBSERVED
             + 0.9·fin_stress_i(t)                                         latent AR(1), seen only through noisy proxies
             + γ · P_i(t)                                                  CONTAGION from disrupted upstream
             + 1.4·country_shock_c(t)                                      common regional shocks
P_i(t) = 1 − Π_j (1 − pass_ij · a_j),  a_j = max(act_j(t−1), 0.7·act_j(t−2), 0.4·act_j(t−3))   (buffers drain over time)
pass_ij = dependency_share_ij × (0.5 if an alternative source exists)
```

Disruptions last 1–6 months and cascade multi-hop through the process itself. Base rate is about 1.1% of suppliers per month (~3.3% with a new incident in the next 3 months). With the default γ=7, about 24% of incident starts are contagion-caused. Raw files are deliberately dirty: ID case/whitespace, 5 spellings of country, percent-vs-fraction units, duplicates, orphan IDs, self loops, negative spend, missing tiers, late corrections. The simulator writes a manifest of every injected defect, and a test checks the silver quarantine matches it exactly.

Ground truth (latents, causes) goes to `data/sim_truth/` and is used only by the clearly-labelled `ORACLE_sim_only` row and by tests.

## Pipeline

```
data/external (real)  ─┐
simulate  → data/raw ──┴→ BRONZE  parquet, untouched + lineage (_source_file, _ingest_run, _row_hash)
                          SILVER  typed, ISO3 countries, deduped, orphans/self-loops/dupes → _quarantine,
                                  percent→fraction, signal VERSIONS kept with the month each became usable,
                                  tier derived from graph, 24 DQ checks (critical ones abort the run)
                          GOLD    point-in-time supplier×month feature panel + labels → scores, graph, products
                                  evaluate (baselines, ablations, CIs, leakage audit)  →  reports/
                                  FastAPI + dashboard over gold
```

Gold features (≈45): own signals (windows, trends, missingness, incident history), static country/sector, **graph structure** (supplier count, sole-source, HHI concentration, criticality = dependency-weighted downstream value-at-stake, upstream country risk) and **graph contagion** (upstream distress through 1 and 2 hops, max, noisy-OR, trend, and a fixed-point noisy-OR *propagated risk*). A placebo feature (customers' distress – wrong causal direction) is included to check the model doesn't credit a graph feature just for being a graph feature.

## Results (seed 42, γ=7; test months 50–56, 14,000 supplier-months, 427 positives)

| model | ROC-AUC | PR-AUC [95% CI] | precision @ top 5% | lift @ 5% |
|---|---|---|---|---|
| base rate | 0.500 | 0.030 | 0.019 | 0.6× |
| heuristic: own distress composite (no learning) | 0.651 | 0.054 | 0.071 | 2.3× |
| heuristic: noisy-OR propagated risk (no learning) | 0.666 | 0.063 | 0.084 | 2.8× |
| logistic, country/static only | 0.602 | 0.048 | 0.053 | 1.7× |
| logistic, own signals + static | 0.708 | 0.081 | 0.106 | 3.5× |
| boosted trees, own + static (**best graph-free**) | 0.705 | 0.078 [0.059, 0.105] | 0.099 | 3.2× |
| + graph structure | 0.717 | 0.082 [0.062, 0.112] | 0.107 | 3.5× |
| **+ graph contagion (main)** | **0.735** | **0.093 [0.070, 0.125]** | **0.123** | **4.0×** |
| oracle (uses simulator latents; not attainable) | 0.766 | 0.127 | 0.183 | 6.0× |

CIs are 200-replicate bootstrap resampling *suppliers* (clusters), not rows. Paired difference, main minus graph-free: PR-AUC +0.0155, CI [+0.005, +0.027]; main minus structure-only: +0.0108, CI [+0.004, +0.019]. Full tables, ablations, calibration, cold-start, DQ and leakage results are regenerated into `reports/REPORT.md` by every run (a committed snapshot of the seed-42 run is in `docs/RESULTS_seed42.md`).

### The experiment that matters: does the graph help only when contagion exists?

`python -m srg.cli sweep`: new simulated worlds (3 seeds each) at four contagion strengths, γ=0 being the **no-contagion control**. Mean test PR-AUC:

| γ | contagion share of starts | boosted, own+static | + structure | + contagion | gain vs structure (sd) | seeds where contagion model wins |
|---|---|---|---|---|---|---|
| 0 (control) | 0% | 0.069 | 0.068 | 0.068 | −0.0003 (0.0018) | 1/3 |
| 3.5 | 9% | 0.076 | 0.080 | 0.080 | +0.0005 (0.0018) | 1/3 |
| 7 | 27% | 0.084 | 0.083 | 0.096 | +0.0121 (0.0124) | 3/3 |
| 10 | 39% | 0.097 | 0.097 | 0.117 | +0.0207 (0.0049) | 3/3 |

With no contagion the graph features add nothing (as they should); the gain scales with contagion strength.

## What these numbers do and do not show

**They show:** the pipeline and evaluation are leak-free by construction and by test; graph-propagated features recover a real portion of contagion risk when it exists in the data (≈30–40% of the gap between the graph-free model and the oracle at γ=7); the method does not hallucinate gains when contagion is absent; an analyst queue of the top 5% (100 suppliers/month) would hit ≈12% incident rate versus 3% at random.

**They do not show:**
- Anything about real-world accuracy. The effect size is a function of my simulator constants (γ, pass-through, signal noise). Whether real supply networks have contagion this strong, and whether signals like these exist, is exactly what this project cannot tell you. The hand-set weights in the distress composite were written with knowledge of how the simulator generates signals (not fitted on test), so even the heuristics are not independent of it.
- Absolute performance is modest on purpose: ROC-AUC ≈ 0.73, PR-AUC ≈ 0.09 at a 3% base rate. The ceiling with perfect knowledge of the latents in this world is only 0.77 / 0.13.
- Most single-run differences are small and sit inside the CIs. Seed-to-seed spread of PR-AUC across simulated worlds (0.05–0.09 in the control) is larger than the effect at γ=3.5. The bootstrap captures test-sample noise, not training or world variance, which is why the sweep exists.
- Group ablations in one run are noisy (e.g. removing "static country/sector" barely moves PR-AUC; permutation importance says own signals ≫ the rest).
- Cold-start (20% of suppliers never seen in training, 84 positives): contagion model ROC-AUC 0.694 vs 0.674, but PR-AUC 0.085 vs 0.093. That is within noise; I make no claim about cold-start improvement.
- Platt scaling on validation did *not* improve calibration (ECE 0.0056 raw vs 0.0075). Raw probabilities are already close to calibrated here, and the served probability uses the raw-then-Platt recipe regardless. It should be revisited, not trusted as-is.
- The product "revenue exposed" figure is `revenue × Σ bom_share × P(3m)`. It is an exposure heuristic, not a loss forecast (no duration, severity or substitution model).

## Leakage controls (all run on every evaluation; see `src/srg/leakage.py`)

1. **Purged temporal split**: train t 6–35, val 39–46, test 50–56; the 3-month label horizon never overlaps the next split (asserted).
2. **Point-in-time features**: a record is usable only from the month it was ingested (first-seen version; later corrections ignored), incidents from their start month, edges from `valid_from`. A test deletes all later data, rebuilds, and requires identical rows (max diff 0.0).
3. **Label-shuffle null**: training on within-month shuffled labels gives test ROC-AUC ≈ 0.52 (3 shuffles: 0.48–0.55).
4. **Canary**: a deliberately leaky feature is added; the univariate scan flags it (AUC 0.88) and the model jumps to 0.91 — i.e. the harness can see leakage, and the real features max out at univariate AUC 0.68.
5. **No simulator latents** in silver/gold (name scan), enforced in tests.
6. Random vs temporal split is reported for information (here: no optimism, 0.734 vs 0.735). I expected a gap; there isn't one, and I report that as found.

## Run it

Python ≥ 3.11. `uv` is used below, but plain `pip install -e ".[dev]"` works.

```bash
make setup      # venv + deps
make run        # simulate → bronze → silver → gold → evaluate + leakage audit → score   (~4 min on 8 cores)
make test       # 30 tests (~75 s); builds a small world end to end
make serve      # http://127.0.0.1:8000  (dashboard at /, OpenAPI at /docs)
make sweep      # contagion sweep incl. control (~15 min)
make docker     # builds the image; the container runs the pipeline on first start, then serves
```

| endpoint | purpose |
|---|---|
| `GET /health`, `/about` | liveness, data provenance caveat |
| `GET /suppliers?tier=&country=&min_p=&limit=` | ranked 3-month disruption probabilities, graph uplift vs. a no-contagion version of the same model |
| `GET /suppliers/{id}` | detail, direct suppliers/customers, strongest upstream drivers |
| `GET /graph/{id}?depth=` | upstream neighbourhood + customers for drawing |
| `POST /simulate {"shocked": [...]}` | what-if: force suppliers to fail, propagate downstream, per-product impact |
| `GET /products` | BOM exposure |
| `GET /metrics`, `/leakage` | the evaluation JSON |

## Layout

```
src/srg/  simulate.py pipeline.py dq.py features.py graph.py models.py evaluate.py leakage.py score.py experiments.py api.py cli.py report.py
scripts/fetch_external.py   data/external/ (committed real snapshot)   dashboard/index.html   tests/   Dockerfile   Makefile
```

## Known limitations / next steps

- Real supplier links are proprietary; plugging in real data means replacing `simulate.py` with extractors that write the same five raw files — everything downstream is unchanged.
- Dependency shares for later-appearing links are not renormalised (the share at creation time is used).
- Edges are static after creation (no deletion); signals have a fixed monthly cadence.
- HGB hyper-parameters come from a 2-point grid on validation PR-AUC; no wider search was run to avoid tuning to the simulator.
- Single focal buyer; no demand-side or price effects; incidents are assumed to be logged accurately and at the start month.
