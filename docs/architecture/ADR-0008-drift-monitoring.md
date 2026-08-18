# ADR-0008 — Prediction logging and PSI drift monitoring

## Status
Accepted

## Context
The API served predictions and retained nothing. Three questions were therefore
unanswerable: what was predicted for a given applicant, whether the model is
still operating on data resembling its training set, and how the service is
performing. The `/metrics` endpoint existed but returned zeros because the table
it queries was never written to.

Detecting whether the model has become *inaccurate* would require ground truth.
For credit default that arrives one to two years after the prediction, and this
project has a single static historical snapshot. No ground truth will ever be
available here.

## Decision
1. **Every prediction is logged** to `monitoring.prediction_log`: the request,
   the engineered feature vector, the output, the latency, and a status.

2. **The logged feature vector is the engineered one**, not the raw request.
   Drift must be measured on what the model actually consumed. This required
   adding `BasePipeline.build_feature_frame()`, which runs `clean()` and
   `feature_engineering(fit=False)` and returns the frame without predicting.

3. **Logging can never break a prediction.** Every function in
   `prediction_logger.py` catches all exceptions, prints to stderr and returns
   a boolean. A logging failure must not turn a successful prediction into a 500.

4. **PSI on engineered features, with a reference frozen at training time.**
   `build_reference` computes quantile bin edges and reference shares from the
   train split and stores them in `reference_profile.json` next to the model.
   Recomputing edges from current data would make PSI structurally always zero.

5. **Thresholds 0.10 (WARN) and 0.25 (ALERT)**, the industry convention, matching
   `DRIFT_PSI_WARN` and `DRIFT_PSI_ALERT` in `.env`.

6. **Batch job, not streaming.** `run_drift_check` runs over a time window on
   demand or on a schedule. Streaming drift detection would add substantial
   infrastructure for a metric that is not actionable minute to minute.

7. **A minimum of 100 rows in the window**, below which the check exits and
   writes nothing. PSI over ten bins on a handful of rows is noise, and a
   misleading report is worse than no report.

8. **Zero bin shares are replaced with 1e-6.** The PSI formula divides by the
   reference share and takes a logarithm of the ratio; an exact zero produces
   infinity. The substitution keeps the value finite and contributes negligibly.

9. **Outer bin edges are open (±infinity).** A live value beyond the training
   range would otherwise fall outside all bins and be silently dropped from the
   count — precisely the drift most worth detecting.

10. **Middleware assigns a request id, times every request, and emits one JSON
    log line per request.** Structured lines can be parsed by a log aggregator;
    prose cannot.

11. **`simulate_drift.py` ships as a demonstration tool**, so the detector can be
    shown firing rather than merely claimed to work.

## Alternatives considered
- **Logging the raw request instead of the engineered vector.** Simpler, but
  measures the wrong thing: drift in imputed income after the median is applied
  is what affects the model, not drift in what was submitted.
- **Reconstructing the vector from `ExplanationResult.all_contributions`.** Would
  have avoided a pipeline change, but only works when an explanation was
  requested, and is indirect.
- **Recomputing bin edges per check.** Would make both distributions uniform by
  construction and PSI always zero.
- **Equal-width rather than quantile bins.** Simpler, but on a skewed feature
  almost everything lands in one bin and changes within it are invisible.
- **KL divergence or a Kolmogorov-Smirnov test instead of PSI.** Both are
  defensible. PSI is the convention in credit risk, has widely understood
  thresholds, and is symmetric in a way KL divergence is not.
- **Asynchronous or queued logging.** Lower latency impact, but adds a queue and
  a worker for a write that takes a few milliseconds.

## Consequences
- **This measures input drift only.** It is an early warning that something may
  be wrong, not a measurement that the model has become less accurate. Without
  ground truth, performance decay is undetectable here. This is the single most
  important limitation of the monitoring layer and is repeated in
  `architecture.md`.
- Logging adds a synchronous database write to every prediction — a few
  milliseconds. Acceptable at this scale; a queue would be the fix at higher
  volume.
- `prediction_log` grows without bound. There is no retention policy. The same
  tension as bronze in ADR-0003: partitioning by date or archiving would be
  needed in production.
- Binary and near-constant engineered features (`income_missing`,
  `has_any_delinquency`) collapse to two or three bins after duplicate edges are
  removed. Their PSI is coarser than for continuous features. Correct, but worth
  knowing when reading a report.
- The reference profile must be rebuilt whenever the model is retrained. It is a
  separate command rather than part of `train.py`, so it can be forgotten.
  Folding it into `run_training` would be a small improvement.
- Drift is computed against the *train* split. Using validation or test would be
  equally defensible; train was chosen because it is what the model actually
  fitted.

## Date
2026-08-14