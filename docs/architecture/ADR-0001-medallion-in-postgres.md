# ADR-0001 — Medallion Architecture in PostgreSQL

## Status
Accepted

## Context
The platform needs a data storage strategy that separates raw ingestion from
cleaned data from analytics-ready data. The source is a single CSV file today
but the design should survive additional data sources in future domains
(fraud, defence, manufacturing).

## Decision
Implement a three-layer medallion architecture (bronze/silver/gold) inside a
single PostgreSQL instance using schemas to separate the layers.

## Alternatives Considered
- A data lake with schema-on-read: more flexible but adds infrastructure
  complexity not justified for a single-node portfolio project.
- A single wide table: simpler but mixes raw and transformed data, making
  debugging and reprocessing harder.

## Consequences
- More ETL code required to move data between layers.
- Single-node scale limit — cannot handle truly large datasets.
- But: clear separation of concerns, easy to reprocess from bronze if
  cleaning logic changes, stable interface for the ML pipeline and dashboard.

## Date
2026-07-24