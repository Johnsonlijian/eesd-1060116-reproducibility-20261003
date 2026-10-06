# Analysis record

The packaged JSON/CSV files are the derived outputs used by the revised manuscript. The FEMA evaluation contains 22 component-1 far-field records at two intensity tiers (44 record/tier rows), and the metric-sensitivity summary applies the five stated weight sets to the same component table. The two IDA summaries add 22 records x 13 intensity levels x 5 conditions at each height (about 2,860 nonlinear analyses in total).

The IDA files report empirical collapse probabilities and finite-capacity lognormal summaries. Records that do not collapse within the tested intensity range remain in the empirical curves; the finite-capacity median and beta are comparative descriptors for this record set, not population estimates with confidence intervals. The 9-story analysis uses the height-specific frozen design value alpha_design = 0.36249.

To rerun the post-processing from the package root:

    python reproducibility/postprocess_metric_sensitivity.py
    python reproducibility/plot_metric_sensitivity.py

To rerun the controller or IDA analyses, point the scripts to the project-controlled benchmark and record locations with `EESD_PROJECT_ROOT`, `EESD_OHTORI_DIR`, and `EESD_FEMA_DIR`, set `EESD_SOURCE_ROOT` when importing a separate source tree, and set `EESD_OUTPUT_DIR` to the desired derived-output folder. The package does not contain raw FEMA records or the original Ohtori benchmark definition files, so a fresh rerun requires those external inputs and their provider permissions.

OpenSeesPy version used for the reported runs: 3.8.0. The local interpreter path is intentionally not embedded in the package.