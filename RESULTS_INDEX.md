# Results index and provenance boundaries

The files below are intentionally separated by evidentiary role. Manuscript numbers must be read from the **canonical** files; exploratory or smoke outputs are retained only when they document a reproducible diagnostic and are labelled as such.

| File(s) | Role | Scope used in the manuscript |
|---|---|---|
| `eesd_ohtori_control_results_fema.json`, `eesd_ohtori_control_fema_rows_full_2026-09-27.csv`, `eesd_fema_full_2026-09-27.json`, `eesd_fema_components_full_2026-09-27.json`, `component_metrics_full_2026-09-27.json` | **Canonical 3-story FEMA set** | 22 records x 2 tiers = 44 rows; Figures 3, 4, 6, 9 and Sections 5.1, 5.2, 5.5. |
| `eesd_ohtori_control9_deepsweep.json` | **Canonical 9-story multi-height/deep-yield set** | 10 canonical + 13 forced-deep cases = 23 rows; Section 5.4 and 5.7. The interval 3.960--6.903% contains six rows. |
| `eesd_ohtori_control9_results.json` | Supporting 9-story baseline snapshot | Earlier 10 canonical + 5 deep-yield subset; retained for provenance and comparison, not used for the 23-case interval count. |
| `eesd_ohtori_control9_smoke.json` | Smoke/diagnostic output | Two-case execution check only; not manuscript evidence. |
| `eesd_semiactive_mr_fullscale.json`, `eesd_semiactive_mr_fullscale_rows3.csv`, `eesd_semiactive_mr_fullscale_rows9.csv` | **Canonical full-scale MR transfer test** | Published Yang et al. (2002) 20-ton device; 44-case 3-story and 12-case 9-story summaries; Figure 6 and Section 5.5. |
| `eesd_ida_fragility_3story_2026-09-30.json`, `eesd_ida_fragility_9story_2026-09-30.json` | **Canonical IDA outputs** | 22-record collapse-limit summaries for Figures 8 and 9. |
| `eesd_ida_fragility.json`, `eesd_ida_fragility9.json` | Figure-compatible aliases | Same IDA result family used by the figure source scripts; retained as aliases for script compatibility. |
| `eesd_ohtori_control_results.json` and `eesd_softening_sensitivity.json` | Supporting base-record/softening checks | Separate 11-case harness and constitutive-sensitivity diagnostics; not substitutes for the FEMA canonical set. |
| `eesd_nearfault_pulse.json` | Supporting synthetic pulse probe | 21 synthetic pulses; Section 5.8 boundary analysis, not recorded near-fault evidence. |

The previously generated three-case exploratory MR file and its rows were removed from the release because its scaled laboratory-device assumptions were superseded by the full-scale Yang et al. transfer test. This prevents a 6-case exploratory result from being mistaken for the 44-case canonical MR analysis.

`ductility_proxy` in legacy CSV headers is a drift-based response proxy, not a measured material ductility; `anova_*` and `ss_*` fields are descriptive variance shares, not inferential ANOVA tests. See `SCHEMA.md`.
