# ADR-0002 — Star Schema for the Gold Layer

## Status
Accepted

## Context
Silver produces one clean, typed row per applicant. Gold needs to serve two
different consumers: the ML pipeline, which wants a stable, model-ready
feature table, and reporting, which wants to slice outcomes by category —
income band, delinquency severity, time — without repeating that categorical
logic in every query.

A flat table (one wide row per applicant, same shape as silver but cleaner)
would satisfy the ML pipeline but push all categorisation logic into every
report and dashboard query separately, and offers no way to preserve history
when an applicant's attributes change.

## Decision
Structure gold as a star schema: a central fact table,
`gold.fact_credit_assessment` (grain: one row per applicant per snapshot
date), surrounded by dimension tables — `dim_date`, `dim_borrower`,
`dim_income_band`, `dim_utilisation_band`, `dim_delinquency_profile`, and
`dim_model`. The fact table stores small integer foreign keys rather than
repeated category labels.

Two further decisions within this:

**`dim_borrower` uses Slowly Changing Dimension Type 2.** When a borrower's
age band or dependents band changes, the old row is closed (marked not
current, given an end date) and a new row inserted, rather than the old value
being overwritten. This preserves what a borrower's attributes were at the
time of any historical assessment.

**`dim_delinquency_profile` is a junk dimension** — every combination of four
boolean flags (recent lates, moderate lates, severe lates, unknown/sentinel
code) is pre-generated as its own row with a severity label, rather than
storing four separate boolean columns directly on the fact table.

## Alternatives considered
- **One flat, wide gold table.** Simpler, and adequate for the ML pipeline
  alone, but every report needing "default rate by income band" would need to
  repeat the banding logic inline, and there would be no way to track
  attribute history for a borrower.
- **Overwriting borrower attributes in place (SCD Type 1).** Simpler than
  Type 2, but destroys history — a decision made 18 months ago could no
  longer be examined against the borrower's attributes as they actually were
  at that time.
- **Four separate boolean columns on the fact table instead of a junk
  dimension.** Avoids a join, but scatters the severity-label logic into every
  query that needs it, rather than defining it once.

## Consequences
- Every fact row requires resolving five foreign keys at load time; a missing
  dimension entry produces a load failure rather than a silently incomplete
  row, by design — the fact loader asserts no null foreign keys and fails
  loudly rather than writing incomplete data.
- Reporting queries become two- or three-table joins rather than single-table
  scans, in exchange for categorical logic living in exactly one place.
- The star schema does not feed the model directly with category labels — the
  ML pipeline reads raw values from a view over the fact and borrower tables,
  and engineers its own features independently. Bands exist for reporting,
  not for the model.

## Date
2026-07-28