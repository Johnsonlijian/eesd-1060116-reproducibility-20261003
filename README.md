# Reproducibility package for EESD manuscript 1060116

This repository contains the code, derived tables, figure-source scripts, and non-sensitive derived outputs supporting the revised manuscript **Actuator-Constraint Handling versus Frequency Retuning in Active Seismic Control: A Two-Height Yielding Steel-Frame Benchmark** (manuscript 1060116, under editorial revision at *Earthquake Engineering & Structural Dynamics*).

The repository is a reproducibility companion. It does not contain the active manuscript, change-marked manuscript, response letter, cover letter, author-consent files, internal audits, raw earthquake records, or the original third-party Ohtori benchmark files.

## Contents

- `RESULTS_INDEX.md` — canonical-vs-supporting result file index and evidence boundaries

- `reproducibility/src/`: benchmark, controller, sensitivity, IDA, MR-device, pulse, and post-processing scripts.
- `reproducibility/figure_sources/`: scripts for the manuscript figures whose derived inputs are retained here.
- `reproducibility/results/`: the checked synthetic-pulse summary.
- `analysis/`: derived JSON, CSV, and NPZ outputs used to check the reported comparisons.
- `figures/`: derived figure PDFs/PNGs for visual inspection.
- `DATASETS_AND_LINKS.csv`: source and redistribution boundary for every external input.
- `REPRODUCIBLE_RUNBOOK.md`: environment, input acquisition, execution order, and known limitations.
- `SCHEMA.md`: meanings and evidence boundaries for retained derived-output field names.
- `requirements.txt`: pinned core Python dependencies.

## Scope and evidence boundary

The retained results cover two planar SAC moment frames, the specified FEMA P695 far-field record set and intensity tiers, a response-level OpenSeesPy bilinear surrogate, bounded active-control comparisons, a published-parameter MR-device transfer, sensitivity analyses, and two-height IDA summaries. The outputs are comparative evidence for this benchmark and do not establish a universal controller ranking, a calibrated full-device deployment, torsional or three-dimensional behavior, or the compiled benchmark's damage indices J7--J10. The source package records where the original trilinear/MEX benchmark and third-party records must be obtained.

The checked results were produced in the project environment. A fresh native OpenSeesPy DLL rerun was not available on the packaging interpreter at release time; the retained derived outputs are therefore distributed with their provenance and the runbook states the rerun boundary explicitly. The NPZ files are derived controller time histories, not raw ground-motion records.

## Reproduction

Read `REPRODUCIBLE_RUNBOOK.md` before running any script. External inputs are not redistributed and must be acquired under their original provider terms. Set `EESD_PROJECT_ROOT`, `EESD_OHTORI_DIR`, `EESD_FEMA_DIR`, `EESD_SOURCE_ROOT`, and `EESD_OUTPUT_DIR` for a local rerun. The post-processing scripts can be run directly on the retained derived outputs.

## License and citation

Original code and derived outputs in this repository are released under the MIT License. Third-party benchmark definitions, ground-motion records, and device parameters are excluded and remain subject to their own licenses and providers. Cite this repository when reusing the code or derived outputs. An archival DOI release has not yet been minted; the archival citation can be added to `CITATION.cff` when it becomes available.
