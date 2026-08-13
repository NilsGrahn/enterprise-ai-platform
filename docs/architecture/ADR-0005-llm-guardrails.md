# ADR-0005 — LLM guardrails for narrative generation

## Status
Accepted

## Context
The platform produces a probability and a set of SHAP contributions. A credit
officer needs a readable assessment note, and under ECOA/Regulation B a declined
applicant must receive specific principal reasons.

A language model can write that note, but introduces risks that are unacceptable
in a regulated decision-support system: it may fabricate figures, invent reason
codes, state a decision it is not permitted to make, or reference protected
characteristics. Its output is also non-deterministic, so it cannot be relied on
for anything that must be reproducible.

## Decision
1. **Separate services.** SHAP explanation (`explain-service`) and narrative
   generation (`llm-service`) are independent. The maths is testable
   deterministically; the LLM can be switched off without affecting predictions
   or explanations.

2. **Structured payload only.** The LLM receives exactly one JSON object built
   by `build_model_output()` — model identity, prediction, population baseline,
   ranked contributions with reason codes, and data quality flags. No DataFrame,
   no PII, no free text from the request. This makes every prompt auditable and
   removes prompt-injection surface.

3. **Deterministic reason codes.** Codes come from a hardcoded feature-to-code
   mapping, not from the model. The LLM writes prose around codes it is handed.

4. **Post-hoc validation.** Output is parsed, validated against a pydantic
   schema, checked so that every reason code used was supplied, and checked so
   that every number appearing in the prose exists in the payload.

5. **One retry, then a deterministic fallback.** A validation failure triggers a
   single retry with the specific error appended. If that also fails, a template
   report built from the same payload is returned with `confidence: low` and
   `llm_status: fallback`. The system degrades; it does not fail.

6. **`llm_status` is always present**, valued `ok`, `fallback` or `disabled`, so
   consumers can display provenance.

7. **Circuit breaker and offline client.** Repeated transport failures open a
   breaker rather than continuing to call a dead service. `NullLLMClient` is used
   when `LLM_ENABLED=false` and in every test — no test calls the real API.

## Alternatives considered
- **Let the LLM produce reason codes.** Rejected: reason codes are a compliance
  artifact drawn from a controlled vocabulary reviewed by legal, not something to
  improvise per request.
- **Trust the system prompt alone.** Prompt instructions reduce bad output but do
  not guarantee it. Mechanical post-hoc validation does, for the specific failure
  modes it covers.
- **Fine-tune a model on approved notes.** Higher quality and more consistent
  wording, but far more effort, and it does not remove the need for validation.
- **Fail the request when the LLM fails.** Rejected: the narrative is
  supplementary. Losing it should not cost the officer the prediction and the
  explanation.
- **Merge explanation and narrative into one service.** Rejected: the SHAP maths
  would then only be testable through non-deterministic output.

## Consequences
- Narrative generation costs an API call and adds latency, so it is opt-in per
  request (`include_narrative`, default false in the Phase 5 API).
- The number validator permits single digits 0-5 as incidental prose values,
  since narratives legitimately contain small counts and ordinals. A fabricated
  single-digit statistic could therefore pass. The check targets invented rates,
  amounts and comparisons, which are effectively never single digits.
- Validation can reject output that was actually fine (a number expressed in a
  representation the normaliser does not generate). The cost is a fallback
  report, which is acceptable; the reverse error is not.
- Adding an engineered feature now requires an entry in both `DISPLAY_NAMES` and
  `REASON_CODES`. Both raise on a missing entry rather than falling back, so the
  omission surfaces at startup.
- `age` is mapped to a reason code phrased as length of credit-relevant history.
  Age is a protected characteristic under ECOA; a production deployment would
  need a fairness review and would likely remove it from the feature set. This is
  recorded as a known limitation rather than resolved here.

## Date
2026-08-12