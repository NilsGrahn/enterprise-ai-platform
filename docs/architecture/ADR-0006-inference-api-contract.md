# ADR-0006 — Inference API contract

## Status
Accepted

## Context
The trained model has to be reachable by other systems — a dashboard, a batch
scoring job, eventually a loan origination system. That requires a network
interface with a stable contract, input validation at the boundary, and honest
reporting of its own readiness.

The interface also has to be domain-agnostic: no file under `inference-api/`
may contain credit-specific logic, or the platform claim from ADR-0004 fails.

## Decision
1. **FastAPI with pydantic models.** One set of model definitions provides
   request validation, response serialisation, and generated OpenAPI
   documentation. Constraints such as `age` in [18, 110] are declarative and
   produce field-level 422 responses without handler code.

2. **Dataset headers accepted as aliases.** `ApplicantFeatures` uses business
   field names internally but accepts the original competition headers
   (`RevolvingUtilizationOfUnsecuredLines`, etc.) as aliases, with
   `populate_by_name` enabled so both spellings work. A raw source row can be
   posted unchanged.

3. **Sentinel values rejected, not reinterpreted.** Values 96 and 98 in the
   late-payment counts are source sentinel codes meaning "unknown". Silver
   nulls them during ETL because bronze data arrives as-is; an API caller is a
   system that should know what it is sending, so the API returns 422 instead.

4. **Explanation on by default, narrative off.** SHAP costs milliseconds; an LLM
   call costs seconds and money. `include_explanation` defaults to true,
   `include_narrative` to false.

5. **Model, explainer and LLM client loaded once at startup** into a
   module-level `ModelStore`, via FastAPI's lifespan hook. Constructing a
   `TreeExplainer` per request would dominate latency.

6. **A failed load is recorded, not raised.** The service starts, `/health`
   reports `model_loaded: false` with `status: error`, and `/predict` returns
   503. Raising would produce a crash loop diagnosable only from container logs.

7. **Health split three ways.** `/health/live` checks nothing but the process,
   so a database blip cannot trigger restarts. `/health/ready` requires model
   and database. `/health` gives the full picture and returns 503 only when the
   model is missing — `degraded` (model up, database down) still returns 200
   because predictions work.

8. **503 versus 422.** 422 means the request was invalid and will fail again
   identically. 503 means the server cannot serve right now and the same
   request may succeed later.

9. **`/metrics` cached for 10 seconds**, because monitoring systems poll
   frequently and each call runs two aggregate queries.

10. **The LLM is never called on a health check.** `llm_reachable` reports the
    circuit-breaker state from ADR-0005 instead.

## Alternatives considered
- **Flask.** Would require writing validation, serialisation and docs by hand.
- **Loading the model per request.** Simpler and always current, but adds
  seconds of latency to every call.
- **Raising on load failure.** Fails faster, but produces a crash loop rather
  than a diagnosable 503.
- **Coercing sentinels to null in the API.** Convenient, but hides a caller-side
  data bug at exactly the boundary where it should surface.
- **Narrative on by default.** Rejected on cost and latency; most consumers want
  a score and drivers, not prose.

## Consequences
- `inference-api/app` is not an installed package, so the API must be started
  from `inference-api/` or with `PYTHONPATH` set. Phase 8's Dockerfile sets it
  explicitly.
- `total_delinquency_events` is required by the pipeline but is derived, not
  supplied by callers. `to_pipeline_frame()` computes it with the same formula
  as `load_fact_assessment.py`. **This duplicates a domain rule in two places.**
  The cleaner fix is to move the derivation into `CreditPipeline.clean()` so
  training and serving share one definition; that was not done here because it
  would change Phase 3 behaviour and require retraining. Recorded as known
  technical debt.
- Risk band boundaries (0.05 / 0.15 / 0.35) are hardcoded in `band_for()` rather
  than configurable. They are policy, like `DECISION_THRESHOLD`, and arguably
  belong in settings.
- Batch prediction loops rather than vectorising. The 500-item cap bounds the
  worst case; vectorising the non-explained path is available future work.
- There is no authentication. Acceptable for a local portfolio deployment,
  unacceptable for anything real. Recorded in the limitations section of
  `architecture.md`.

## Date
2026-08-12