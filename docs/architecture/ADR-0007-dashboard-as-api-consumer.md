# ADR-0007 — The dashboard is an API consumer, not a model host

## Status
Accepted

## Context
The dashboard needs predictions with SHAP explanations. It runs in the same
repository as the ML pipeline and could import `ml_service`, load the artifact,
and call `pipeline.predict()` directly — avoiding an HTTP round trip and a
running dependency.

It also needs population-level aggregates over 150,000 gold rows for the
portfolio view, which is a different kind of access with different constraints.

## Decision
1. **All predictions go through the inference API over HTTP.** The dashboard
   never imports `ml_service`, `explain_service` or `llm_service`, and never
   loads a model artifact.

2. **Read-only aggregate queries go directly to Postgres**, through
   `data_platform.db`, cached with `@st.cache_data`. Routing aggregates over
   150,000 rows through an HTTP API would add latency for no benefit and would
   require inventing reporting endpoints.

3. **API failures surface as banners, not tracebacks.** Every client call
   returns `(data, error_message)`; the UI must handle the error branch.

4. **The API base URL comes from `INFERENCE_API_URL`** with a localhost
   default, because inside a container `localhost` refers to that container.

5. **Provenance is always displayed** on the analyst screen: model version,
   request id, and latency.

6. **`llm_status` is surfaced visually** — success, warning or info — so a user
   can tell generated prose from a deterministic template.

## Alternatives considered
- **Import the pipeline directly.** Faster and simpler, but the API would
  become an untested side-feature, two model instances could drift apart, and
  the Phase 8 dashboard image does not contain `ml-service/`.
- **Route portfolio aggregates through the API too.** Architecturally purer,
  but would require reporting endpoints that exist only to serve one consumer,
  and would move heavy aggregation into the request path.
- **A pre-built read model for the portfolio view.** Better at scale; not
  justified at 150,000 rows where the queries return in milliseconds.

## Consequences
- The dashboard requires a running API for its core function. This is the point
  — it is what makes the API the tested integration path.
- Two access paths mean two failure modes: the API can be down while Postgres is
  up. The portfolio view degrades partially rather than failing entirely.
- The dashboard duplicates the field-alias mapping from `ApplicantFeatures`.
  Deliberate — it exercises the documented external contract rather than a
  convenience path. It would break if the aliases changed without updating both;
  a shared schema package would fix that at the cost of coupling.
- `st.number_input` cannot produce null, so a zero monthly income is mapped to
  `None` to exercise the missing-income path. Documented in the field's help
  text; a checkbox would be less surprising.
- Scoring-activity and drift panels are empty until Phase 7. They render an
  explanatory message rather than nothing.

## Date
2026-08-13