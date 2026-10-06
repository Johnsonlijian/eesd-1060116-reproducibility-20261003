# Derived-output schema notes

The `analysis/` directory contains machine-readable outputs retained from the checked revision. A few field names preserve the source-analysis schema so that the manuscript tables can be traced without transformation:

- `ductility_proxy` means the drift-based ductility proxy used to rank uncontrolled demand depth. It is descriptive and does not imply a calibrated damage model.
- Keys beginning with `anova_` and fields such as `ss_A_frac`, `ss_B_frac`, and `ss_AB_frac` are descriptive two-factor variance-share calculations. They are not inferential ANOVA tests, do not assume independent records, and do not carry p-values.
- J1--J6 are response/effort metrics retained in some derived files. J7--J10 are source-schema labels only; the open response-level surrogate in this release does not reproduce the compiled benchmark damage indices.
- NPZ files in `analysis/` contain derived control time histories and controller traces (`t`, drift, force, plan, and related arrays). They contain no raw ground-motion acceleration record and are not a redistribution of the external record sets.

The clean manuscript uses reader-facing terminology; these schema notes preserve traceability for the release files without extending the scientific claims.
