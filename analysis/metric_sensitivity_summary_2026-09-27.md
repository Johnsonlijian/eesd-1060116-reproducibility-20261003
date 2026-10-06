# Metric-weight sensitivity (44 FEMA record/tier rows)

The component rerun retains RMS drift, peak drift, peak force, and separate peak acceleration for all four factorial cells. The five weight sets are applied to normalized RMS drift, peak drift, and active-force effort.

## Mean composite score (lower is better)

| weight set | elastic/clip | softened/clip | elastic/in-QP | softened/in-QP | lowest mean |
|---|---:|---:|---:|---:|---|
| primary | 0.6204 | 0.6250 | 0.5927 | 0.6070 | elastic_inqp |
| balanced | 0.6393 | 0.6437 | 0.6006 | 0.6090 | elastic_inqp |
| response_dominant | 0.5810 | 0.5857 | 0.5587 | 0.5761 | elastic_inqp |
| rms_drift_emphasis | 0.5615 | 0.5664 | 0.5448 | 0.5651 | elastic_inqp |
| effort_aware | 0.6268 | 0.6310 | 0.6056 | 0.6207 | elastic_inqp |

## In-QP minus clipping

| weight set | tuning | n | in-QP wins | clipping wins | ties | mean delta |
|---|---|---:|---:|---:|---:|---:|
| primary | elastic | 44 | 32 | 12 | 0 | -0.0278 |
| primary | softened | 44 | 31 | 13 | 0 | -0.0180 |
| balanced | elastic | 44 | 33 | 11 | 0 | -0.0387 |
| balanced | softened | 44 | 34 | 10 | 0 | -0.0347 |
| response_dominant | elastic | 44 | 32 | 12 | 0 | -0.0223 |
| response_dominant | softened | 44 | 30 | 14 | 0 | -0.0097 |
| rms_drift_emphasis | elastic | 44 | 32 | 12 | 0 | -0.0168 |
| rms_drift_emphasis | softened | 44 | 28 | 16 | 0 | -0.0013 |
| effort_aware | elastic | 44 | 32 | 12 | 0 | -0.0211 |
| effort_aware | softened | 44 | 31 | 13 | 0 | -0.0104 |

Peak acceleration is retained in the raw component output but is not inserted into the primary score. The ranking statement is therefore explicitly conditional on this response/effort family.
