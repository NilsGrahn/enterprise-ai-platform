-- Model-facing interface for the credit pipeline.
-- Joins the assessment facts to the borrower dimension so that age and
-- number_of_dependents are available alongside the measures.
--
-- A view is used rather than widening the fact table because age and
-- dependents are borrower attributes, not assessment measures — putting them
-- in the fact table would break the star schema's grain and duplicate data
-- that dim_borrower already versions correctly.

CREATE OR REPLACE VIEW gold.v_credit_assessment AS
SELECT
    f.applicant_id,
    f.snapshot_date_key,
    f.dataset_split,
    f.is_serious_delinquency,

    f.revolving_utilisation,
    f.debt_ratio,
    f.monthly_income,
    f.open_credit_lines,
    f.real_estate_loans,
    f.times_30_59_days_late,
    f.times_60_89_days_late,
    f.times_90_days_late,
    f.total_delinquency_events,

    b.age,
    b.number_of_dependents
FROM gold.fact_credit_assessment f
JOIN gold.dim_borrower b
  ON f.borrower_key = b.borrower_key;