# ADR-0004 — Pipeline abstraction, registry, and the artifact contract

## Status
Accepted

## Context
The platform is intended to serve multiple domains — credit today, potentially
fraud, defence, or manufacturing later. Without a deliberate abstraction,
domain logic spreads into training scripts, evaluation, the API, and the
dashboard, and "supporting a second domain" becomes a rewrite rather than an
addition.

A second, independent pressure is train/serve skew: any transformation learned
from training data (medians, outlier caps, column ordering) must be applied
identically at serving time. When training and serving have separate code
paths, they drift, and the failure is silent — the model returns plausible
numbers that are quietly wrong.

## Decision
1. An abstract base class, `BasePipeline`, defines four abstract methods
   (`clean`, `feature_engineering`, `train`, `predict`) plus a required
   `required_columns` property, and three identity attributes (`name`,
   `target_column`, `source_table`).

2. `run_training()` is a **template method**: concrete in the base class and
   not overridable. It fixes the order of operations — load, validate, clean,
   engineer with `fit=True` on train and `fit=False` on validation, train,
   save artifact, register model. Subclasses supply the contents of steps,
   never the sequence.

3. A registry (`pipelines/__init__.py`) maps pipeline names to classes.
   `ACTIVE_PIPELINE` in `.env` selects which one is live.

4. Every trained model is persisted as an artifact pair: `model.pkl` plus a
   `metadata.json` whose `preprocessing` block holds every value learned from
   training data, and whose `feature_names` list freezes column order.
   Serving reads these; it never recomputes them.

5. Model registration (`gold.dim_model`) is separate from model activation.
   Activation swaps `is_active` inside one transaction, guarded by a partial
   unique index permitting at most one active model per pipeline.

6. Domain data reaches the pipeline through a view (`gold.v_credit_assessment`)
   rather than a raw table, so `load_data()` stays generic and the model's
   permitted inputs are explicit.

## Alternatives considered
- **Separate training scripts per domain.** Simplest to start, but every
  infrastructure change would need repeating per domain, and the versions
  would drift.
- **Configuration-driven pipelines (YAML describing features).** Avoids
  writing Python per domain, but feature engineering quickly outgrows what a
  config file can express, and debugging a config interpreter is harder than
  debugging Python.
- **A scikit-learn `Pipeline` object serialised whole.** Handles train/serve
  consistency well for standard transformers, but ties the design to
  scikit-learn's transformer API and makes the learned values opaque rather
  than inspectable in a JSON file.
- **Recomputing preprocessing at serve time.** Rejected outright: a single
  request has no population to compute a median from, and batch statistics
  would differ from training's.

## Consequences
- Every service depends on `BasePipeline`, never on `CreditPipeline`.
- Adding a domain touches exactly two files: a new `{domain}_pipeline.py` and
  one line in the registry, plus one `.env` change.
- The training protocol cannot be varied per domain, by construction. This is
  intentional and is the main thing the abstraction buys.
- `metadata.json` becomes a hard compatibility surface: changing the shape of
  `preprocessing` invalidates existing artifacts. Model versioning is the
  mitigation.
- Pickle-family serialisation is not portable across very different library
  versions; `hyperparameters` is recorded so a model can be retrained
  identically if a load ever fails.
- A domain whose training genuinely needs a different protocol (say,
  time-series cross-validation instead of a fixed split) would need
  `BasePipeline` extended rather than bypassed.

## Date
2026-08-10