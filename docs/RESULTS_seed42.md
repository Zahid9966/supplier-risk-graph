# Evaluation report (auto-generated)

Task: predict a NEW supplier disruption within the next 3 months, scored monthly. All data below is **simulated** except the country covariates (see README).

| split | months | rows | positives | base rate |
|---|---|---|---|---|
| train | 6-35 | 60000 | 1956 | 0.033 |
| val | 39-46 | 16000 | 590 | 0.037 |
| test | 50-56 | 14000 | 427 | 0.030 |

## Test-set results (95% supplier-clustered bootstrap CI on PR-AUC)

| model | ROC-AUC | PR-AUC [CI] | prec@5% | recall@5% | lift@5% | Brier | ECE |
|---|---|---|---|---|---|---|---|
| base_rate | 0.500 | 0.030 [0.026, 0.035] | 0.019 | 0.030 | 0.61 | 0.030 | 0.002 |
| heur_own_composite | 0.651 | 0.054 [0.044, 0.068] | 0.071 | 0.117 | 2.34 | - | - |
| heur_propagated | 0.666 | 0.063 [0.051, 0.081] | 0.084 | 0.138 | 2.76 | - | - |
| logit_static | 0.602 | 0.048 [0.036, 0.065] | 0.053 | 0.087 | 1.73 | 0.029 | 0.009 |
| logit_own | 0.708 | 0.081 [0.064, 0.107] | 0.106 | 0.173 | 3.47 | 0.030 | 0.010 |
| hgb_own | 0.705 | 0.078 [0.059, 0.105] | 0.099 | 0.162 | 3.23 | 0.029 | 0.009 |
| hgb_own_struct | 0.717 | 0.082 [0.062, 0.112] | 0.107 | 0.176 | 3.51 | 0.029 | 0.007 |
| logit_full | 0.729 | 0.086 [0.066, 0.119] | 0.106 | 0.173 | 3.47 | 0.029 | 0.009 |
| hgb_full | 0.735 | 0.093 [0.070, 0.125] | 0.123 | 0.201 | 4.03 | 0.029 | 0.006 |
| ORACLE_sim_only | 0.766 | 0.127 [0.093, 0.171] | 0.183 | 0.300 | 6.00 | 0.028 | 0.005 |

Paired PR-AUC difference `hgb_full` minus hgb_own_struct: mean +0.0108, 95% CI [+0.0043, +0.0190], positive in 100% of resamples.

`hgb_full` minus `hgb_own` (graph-free): mean +0.0155, 95% CI [+0.0051, +0.0273].

## Ablations (remove one feature group from the main model)

| variant | ROC-AUC | PR-AUC | prec@5% |
|---|---|---|---|
| minus_own_signals | 0.719 | 0.086 | 0.120 |
| minus_static_country_sector | 0.726 | 0.087 | 0.130 |
| minus_graph_structure | 0.727 | 0.082 | 0.110 |
| minus_graph_contagion | 0.717 | 0.082 | 0.107 |
| plus_placebo_downstream | 0.734 | 0.089 | 0.117 |
| full | 0.735 | 0.093 | 0.123 |

Permutation importance (PR-AUC drop, test, 5 shuffles): own_signals +0.0416, static_country_sector +0.0144, graph_structure +0.0145, graph_contagion +0.0120

## Calibration (main model)
Raw: Brier 0.0288, ECE 0.0056. Platt on validation: Brier 0.0288, ECE 0.0075. Constant base-rate Brier 0.0296.

## Cold-start suppliers (20% of suppliers never seen in training; 2800 rows, 84 positives)

- hgb_own: ROC-AUC 0.674, PR-AUC 0.093
- hgb_full: ROC-AUC 0.694, PR-AUC 0.085

## Leakage audit

| check | result | detail |
|---|---|---|
| split_purge_gap | PASS | label horizon 3m; splits {'train': (6, 35), 'val': (39, 46), 'test': (50, 56)} leave no label overlap |
| no_simulator_latents_in_pipeline | PASS | forbidden-name scan over silver+gold columns: clean |
| point_in_time_truncation | PASS | rebuilt features from data truncated at t0=[30, 45]; max abs diff over 8000 rows = 0.00e+00 |
| label_shuffle_null | PASS | trained on within-month shuffled labels, 3 shuffles -> test ROC-AUC 0.517 (runs: 0.522, 0.476, 0.551); should be ~0.5 |
| canary_and_univariate_scan | PASS | max univariate AUC of real features 0.681 (flag >0.85: none); canary univariate AUC 0.881 is flagged; model with canary reaches test AUC 0.908 |
| random_vs_temporal_split | PASS | informational: random row split ROC-AUC 0.734 vs temporal 0.735 (gap -0.002); we report temporal only |

## Data quality (silver)

24 checks, 24 passed. 

| check | severity | observed | expectation | pass |
|---|---|---|---|---|
| suppliers_unique_id | critical | 0 | == 0 duplicates | yes |
| suppliers_country_mapped | critical | 0 | all ISO3 in reference | yes |
| suppliers_spend_positive | critical | 0 | no negatives | yes |
| suppliers_spend_null_rate | warn | 0.017 | <= 0.05 | yes |
| tier_reported_vs_graph_derived_agreement | warn | 0.893 | >= 0.85 (skip-tier edges explain the rest) | yes |
| edges_no_self_loops | critical | 0 | == 0 | yes |
| edges_endpoints_known | critical | 0 | referential integrity | yes |
| edges_customer_share_sum_off | warn | 0.0027 | <= 0.02 | yes |
| edges_share_range | critical | 1.0 | within [0,1] | yes |
| edges_alt_unknown_rate | warn | 0.0467 | <= 0.1 | yes |
| signals_otd_range | critical | 1.0 | within [0,1] after unit fix | yes |
| signals_fin_range | warn | 140.99 | plausible range | yes |
| signals_news_nonneg | critical | 0.0 | >= 0 | yes |
| signals_avail_after_period | critical | 0 | ingestion never precedes the period (point-in-time safety) | yes |
| signals_otd_null_rate | warn | 0.2963 | <= 0.75 | yes |
| signals_fin_score_null_rate | warn | 0.5763 | <= 0.75 | yes |
| signals_news_neg_null_rate | warn | 0.3369 | <= 0.75 | yes |
| signals_price_vol_null_rate | warn | 0.2981 | <= 0.75 | yes |
| signals_monthly_coverage_min | critical | 0.967 | >= 0.9 of suppliers have a record each month | yes |
| incidents_have_supplier | critical | 0 | referential integrity | yes |
| incidents_dates_ordered | critical | 0 | end >= start | yes |
| bom_share_sums_to_one | critical | 0 | per product | yes |
| bom_suppliers_known | critical | 0 | referential integrity | yes |
| graph_depth_finite | warn | 4 | derived max tier <= 6 | yes |

Injected raw defects (simulator manifest): `{"suppliers_missing_tier": 128, "suppliers_negative_spend": 34, "suppliers_dirty_id": 194, "suppliers_duplicate_rows": 5, "edges_duplicates": 12, "edges_orphans": 3, "edges_self_loops": 1, "signals_corrections": 3503, "signals_orphans": 40, "signals_pct_unit_rows": 3359}`

## Contagion-strength sweep (does the graph help only when contagion exists?)

Mean test PR-AUC over seeds [1, 2, 3] (each seed = a new simulated world; gamma = contagion strength, 0 = control).

| gamma | contagion share of starts | hgb_own | hgb_own_struct | hgb_full | full - own (sd) | full - struct (sd) | seeds where full > struct |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.069 | 0.068 | 0.068 | -0.0008 (0.0025) | -0.0003 (0.0018) | 1/3 |
| 3.5 | 0.09 | 0.076 | 0.080 | 0.080 | +0.0040 (0.0027) | +0.0005 (0.0018) | 1/3 |
| 7 | 0.27 | 0.084 | 0.083 | 0.096 | +0.0115 (0.0121) | +0.0121 (0.0124) | 3/3 |
| 10 | 0.39 | 0.097 | 0.097 | 0.117 | +0.0204 (0.0041) | +0.0207 (0.0049) | 3/3 |

