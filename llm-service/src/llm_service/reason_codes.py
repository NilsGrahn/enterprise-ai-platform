class MissingReasonCodeError(Exception):
    """Raised when a feature has no mapped adverse-action reason code."""


REASON_CODES = {
    'revolving_utilisation':    ('R01', 'Proportion of revolving credit in use'),
    'times_90_days_late':       ('R02', 'History of seriously delinquent payments'),
    'debt_ratio':               ('R03', 'Level of existing debt relative to income'),
    'monthly_income':           ('R04', 'Income level relative to obligations'),
    'open_credit_lines':        ('R05', 'Number of open credit accounts'),
    'times_60_89_days_late':    ('R06', 'History of moderately delinquent payments'),
    'times_30_59_days_late':    ('R07', 'History of recent late payments'),
    'total_delinquency_events': ('R08', 'Overall frequency of late payments'),
    'real_estate_loans':        ('R09', 'Number of real estate secured obligations'),
    'age':                      ('R10', 'Length of credit-relevant history'),
    'number_of_dependents':     ('R11', 'Household obligations relative to income'),
    'income_missing':           ('R12', 'Income could not be verified'),
    'monthly_debt_payment':     ('R13', 'Estimated monthly debt service burden'),
    'income_per_dependent':     ('R14', 'Income available per household member'),
    'delinquency_ratio':        ('R15', 'Late payments relative to accounts held'),
    'has_any_delinquency':      ('R16', 'Presence of any delinquency on record'),
    'utilisation_bucket':       ('R17', 'Credit utilisation band'),
}


def code_for(feature: str):
    """Return (code, reason_text) for one feature."""
    if feature not in REASON_CODES:
        raise MissingReasonCodeError(
            f"No reason code mapped for feature '{feature}'. "
            f"Add it to REASON_CODES in llm_service/reason_codes.py."
        )
    return REASON_CODES[feature]


def to_reason_codes(explanation) -> list:
    """Adverse-action reason codes for one explanation, in rank order.

    Only risk-increasing contributions produce a reason code — a factor that
    helped the applicant is not a reason for an adverse decision.
    """
    result = []
    for contribution in explanation.increases_risk():
        code, reason = code_for(contribution.feature)
        result.append({
            'code': code,
            'reason': reason,
            'rank': contribution.rank,
            'feature': contribution.feature,
        })
    return sorted(result, key=lambda r: r['rank'])