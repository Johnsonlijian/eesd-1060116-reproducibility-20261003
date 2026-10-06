# Reproducible runbook

## Environment

- Python 3.12 recommended.
- OpenSeesPy 3.8.0 for nonlinear reruns, with NumPy, SciPy, pandas, and Matplotlib.
- MiKTeX is only needed to compile the separate manuscript package; manuscript TeX files are intentionally not in this repository.
- Run from the repository root so all relative output paths remain portable.

Create a clean environment and install the scientific dependencies required by the scripts. Do not put raw records, the Ohtori source archive, credentials, or private project paths in this repository.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The pinned OpenSeesPy version is the version used for the reported project runs; the packaging host did not provide a loadable native DLL for a fresh rerun.

## External inputs

The scripts expect the original benchmark and record locations through environment variables:

- `EESD_PROJECT_ROOT`: project root used by the scripts when resolving package-relative inputs.
- `EESD_OHTORI_DIR`: licensed Ohtori benchmark definition directory containing the supplied `struct3.m`, `struct9.m`, and related files.
- `EESD_FEMA_DIR`: local FEMA/ATC far-field record directory.
- `EESD_SOURCE_ROOT`: optional source tree for shared project utilities.
- `EESD_OUTPUT_DIR`: writable directory for generated derived outputs.
- `EESD_FIGURE_OUTPUT_DIR`: writable directory for generated figure files; defaults to the package `figures/` directory.

The exact external sources and redistribution status are in `DATASETS_AND_LINKS.csv`. The release does not redistribute these inputs.

## Recommended order

1. Run the benchmark validation scripts in `reproducibility/src/` and inspect the generated plant and modal-validation JSON.
2. Run the FEMA component and primary evaluation scripts. Keep the record and intensity definitions fixed to the source registry.
3. Run `reproducibility/postprocess_metric_sensitivity.py` and `reproducibility/plot_metric_sensitivity.py` for the five reported weight sets.
4. Run the adaptive-MPC, MR-transfer, softening, mechanism time-history, pulse, and IDA scripts only after the base validation passes.
5. Run the scripts in `reproducibility/figure_sources/` to regenerate the corresponding vector figures after pointing them to the intended derived-output directory.
6. Compare regenerated JSON/CSV outputs with the checked files under `analysis/`. Small differences can arise from solver versions, BLAS libraries, record parsing, and the unavailable compiled source model. See `SCHEMA.md` for retained field names and the distinction between descriptive variance shares and inferential tests.

Example post-processing commands:

```powershell
python reproducibility/postprocess_metric_sensitivity.py
python reproducibility/plot_metric_sensitivity.py
```

Set `EESD_FIGURE_OUTPUT_DIR` before running figure scripts when the default package `figures/` location is not desired.

## Checked release outputs

The checked release retains the FEMA component tables, five-weight sensitivity summaries, mechanism sweeps, MR and adaptive diagnostics, synthetic near-fault pulse summary, and two-height IDA summaries. The IDA set contains 22 records, 13 intensity levels, and five controller conditions at each height. The finite-capacity lognormal medians and betas are descriptive summaries for this tested record set, not population estimates with confidence intervals.

## Known limitations

- The open plant is a response-level bilinear Steel01 approximation to the source trilinear/MEX benchmark; it is not a bit-exact replacement and does not reproduce J7--J10 damage indices.
- The yield-adaptive controller is an online linear-model update, not a full nonlinear trajectory MPC.
- The MR branch transfers published device parameters and is not a frame-specific experimental calibration.
- Torsional, out-of-plane, diaphragm-flexibility, and other three-dimensional effects are outside the model.
- A fresh native OpenSeesPy DLL rerun was not available on the packaging interpreter; all retained results and their provenance remain available for audit.
