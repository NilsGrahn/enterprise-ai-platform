# ADR-0010 — Test strategy and CI/CD

## Status
Accepted

## Context
The platform spans seven services and roughly forty files. Verification had been
manual: run a command, read the output. That is slow, so it happens rarely, and
incomplete, because it checks the thing just changed rather than everything that
depended on it.

The architecture also rests on claims that were previously protected only by
discipline: that infrastructure depends solely on `BasePipeline`, that training
and serving apply identical transformations, and that the dashboard never loads a
model.

## Decision
1. **No unit test touches a live database, a real LLM, or a trained artifact.**
   Fixtures build synthetic data and train a small model in-session. A test that
   needs the network fails for reasons unrelated to correctness, and one that
   calls a paid API costs money to run.

2. **A contract test suite over `PIPELINE_REGISTRY`.** Parametrised over every
   registered pipeline, it asserts each subclasses `BasePipeline`, defines its
   three identity attributes, implements all four abstract methods, and does not
   override `run_training` or `load_data`. Any pipeline added in future is
   checked automatically. This is the machine-checkable form of ADR-0004.

3. **The train/serve skew guarantee is tested explicitly.** One test builds a
   frame whose own median differs sharply from the training median and asserts
   the stored value is used. Another asserts column order is identical across
   `fit=True` and `fit=False`.

4. **The SHAP additivity property is a permanent test**, not a one-off scratch
   script. It is the only check that catches the wrong-class-index bug, which
   silently inverts every explanation without raising.

5. **The LLM guardrails are tested with mocks.** Fabricated reason codes,
   fabricated numbers, malformed JSON and transport failures are each simulated,
   and the retry count is asserted so "retry once, then fall back" is pinned
   rather than assumed.

6. **API tests use FastAPI's `TestClient`** with a monkeypatched `ModelStore`, so
   endpoints, validation and status codes are exercised in-process with no
   server, network, artifact or database.

7. **Behavioural rather than numerical assertions.** Tests assert ranges,
   shapes and relationships — a riskier profile scores higher — never a specific
   probability, so they survive retraining.

8. **CI runs lint, then migrations against a real Postgres service container,
   then the unit tests.** The migration step is deliberately an integration
   check rather than a unit test: it proves the SQL applies to an empty database.

9. **Image builds are gated on tests passing** via `needs:`. A failing test
   skips the build job entirely.

10. **CI additionally asserts the dashboard image cannot import `ml_service`**,
    making the ADR-0007 boundary enforced on every commit rather than verified
    once by hand.

11. **The coverage floor starts at 45% rather than 60%.** A gate that fails for
    reasons unrelated to correctness trains people to ignore CI. It should be
    raised deliberately as coverage grows.

12. **CD publishes to GHCR on `main` and on version tags**, using a build matrix
    over the three services, tagged by commit SHA for traceability.

## Alternatives considered
- **Integration tests against a real database throughout.** Higher fidelity, but
  slow, flaky, and requiring database setup wherever the suite runs.
- **Using the real trained artifact in tests.** Would make the suite depend on a
  gitignored file and break on every retrain.
- **Recording real LLM responses and replaying them.** Better fidelity than
  mocks, but the responses go stale and the recording still costs money.
- **Testing exact prediction values.** Would break on any retraining, and tests
  the model rather than the code.
- **A single CI job doing everything.** Simpler, but loses the explicit gate
  between "tests passed" and "images built", which is the pipeline's main value.

## Consequences
- Unit tests do not exercise real SQL against real data. A logic error in a
  migration would only surface in the migration step or at runtime.
- The synthetic fixtures must be kept in step with the real schema. If a column
  is added to gold, the fixtures need updating or tests will fail for a reason
  unrelated to the change.
- Mocked LLM tests verify the guardrails, not the prompt's actual quality. Real
  output quality remains a manual judgement.
- The suite trains a small model per session, so it is not instant. Session
  scope keeps this to once per run.
- `pyproject.toml` needs `pythonpath = ["inference-api"]` because
  `inference-api/app` is not an installed package — the same friction noted in
  ADR-0006.
- The coverage floor is currently low enough not to be a meaningful gate. It is
  a placeholder to be tightened.

## Date
2026-08-16