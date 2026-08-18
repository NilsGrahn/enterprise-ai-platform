# ADR-0011 — Verifying the platform claim

## Status
Accepted

## Context
Since ADR-0004 the project has claimed that infrastructure depends only on
`BasePipeline` and that domain knowledge is confined to
`ml-service/src/ml_service/pipelines/`. Until Phase 10 that claim rested on
discipline, a grep for the string "credit" outside the pipelines folder, and a
contract test asserting registry entries satisfy the interface.

None of those prove the claim. A leak need not contain the word "credit", and a
contract test verifies the interface without exercising a second implementation.

## Decision
1. **A second pipeline, `DummyPipeline`, is built and kept permanently.** It
   differs from `CreditPipeline` in source table, feature count, algorithm and
   preprocessing — every dimension the abstraction is meant to tolerate.

2. **It reads `gold.fact_credit_assessment` rather than the view**, so
   `load_data()` is exercised against a differently-shaped source.

3. **It uses `DecisionTreeClassifier`, not `LogisticRegression`** as originally
   specified, because `shap.TreeExplainer` does not support linear models. This
   is a workaround, not a fix, and is recorded as a limitation below.

4. **It reuses two existing feature names** so that `DISPLAY_NAMES` and
   `REASON_CODES` have entries. Also a workaround.

5. **The test is: swap `ACTIVE_PIPELINE`, verify the API, dashboard, monitoring
   and containers work with no edits outside `pipelines/`.**

6. **`dummy_pipeline.py` is kept, not deleted.** The Phase 9 contract tests are
   parametrised over `PIPELINE_REGISTRY`, so it becomes a permanent regression
   test that CI runs on every commit.

## Result
Working unmodified: `train.py`, `evaluate.py`, `base_pipeline.py`,
`artifacts.py`, `registry.py`, all of `inference-api/`, all of `dashboard/`, all
of `monitoring/`, all of `infrastructure/`, all of `.github/`, and all of
`tests/` — with test coverage extending to the new pipeline automatically.

Total change: one new file, one registry line, one environment variable.

## What the test exposed
Two credit-specific assumptions in infrastructure:

**1. `ShapTreeExplainer` assumes a tree model.** `shap.TreeExplainer` supports
tree ensembles only. A linear-model pipeline fails at explainer construction and
`ModelStore` records the failure, leaving the API reporting
`model_loaded: false`. `BasePipeline` places no constraint on model type —
`TrainingResult.model` is typed `Any` deliberately — so this is infrastructure
assuming something the contract does not.

*Proposed fix:* a factory selecting `TreeExplainer`, `LinearExplainer` or
`KernelExplainer` by model type. The `BaseExplainer` abstract class already
exists; only the factory is missing. A smaller mitigation is to make
`ModelStore.load()` tolerate explainer construction failure and degrade to
predictions without explanations, matching the `llm_status: disabled` pattern
from ADR-0005.

**2. `DISPLAY_NAMES` and `REASON_CODES` are keyed by credit feature names**, and
both raise on a missing entry rather than falling back — deliberately, per
ADR-0005, because a raw variable name must never reach a credit officer and
reason codes are a controlled compliance vocabulary. But both dictionaries live
in infrastructure.

*Proposed fix:* move both onto `BasePipeline` as properties, with each concrete
pipeline supplying its own, and have the infrastructure read them from the
pipeline.

Both fixes are deferred. Recording them is more useful than an unqualified pass.

## Alternatives considered
- **Grep alone.** Cheap, but only catches a literal string, not an assumption.
- **A fully realistic second domain.** Stronger, but requires a second dataset,
  schema and ETL — weeks of work to test one property.
- **Deleting the dummy after testing.** Would lose the permanent regression
  value the contract tests provide for free.
- **Making the dummy identical to credit except in name.** Would pass trivially
  and prove nothing.

## Consequences
- `PIPELINE_REGISTRY` permanently contains a non-production pipeline. Its
  docstring states its purpose. It cannot be activated accidentally without an
  explicit `.env` change.
- Every contract test now runs twice, roughly doubling that file's runtime —
  negligible, and the coverage is the point.
- The explainability limitation is now documented rather than latent. A future
  domain using a non-tree model would hit it immediately, with a clear record of
  why and how to fix it.
- Two active models coexist in `dim_model`, one per pipeline. Correct: the
  partial unique index permits one active model *per pipeline*.

## Date
2026-08-18