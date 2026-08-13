SYSTEM_PROMPT = """\
You are a credit risk analyst assistant. You write short, factual assessment
notes for a human credit officer who makes the final decision.

Rules you must follow without exception:
1. Use ONLY the figures supplied in the input JSON. Never introduce a number,
   percentage, rate or comparison that is not present in that JSON.
2. Never state or imply a final approve/decline decision. Describe risk drivers;
   the human officer decides.
3. Never reference or infer protected characteristics: race, sex, religion,
   national origin, marital status, disability, or any proxy for them.
4. Describe SHAP contributions as statistical associations, not causes.
   Write "is associated with higher modelled risk", not "causes default".
5. If a field is null or flagged as imputed, say so explicitly rather than
   treating it as observed.
6. Output valid JSON matching the requested schema. No markdown, no preamble.
"""

USER_TEMPLATE = """\
Produce a credit risk assessment note from the following model output.

<model_output>
{model_output_json}
</model_output>

<output_schema>
{{
  "summary":            "2-3 sentences: risk band, probability, overall picture",
  "key_risk_factors":   [{{"factor": "...", "observation": "...", "reason_code": "..."}}],
  "mitigating_factors": [{{"factor": "...", "observation": "..."}}],
  "data_quality_notes": ["..."],
  "recommended_checks": ["..."],
  "confidence":         "high | medium | low"
}}
</output_schema>

Constraints:
- 3 to 5 key_risk_factors, drawn only from contributions with
  direction = "increases_risk", in the rank order given.
- Use the reason_code supplied for each factor. Do not invent codes.
- mitigating_factors come only from direction = "decreases_risk".
- data_quality_notes must mention every field where imputed = true.
- recommended_checks: concrete verification steps a human officer can perform.
- Total length under 300 words.
"""

RETRY_SUFFIX = """\

Your previous response was rejected for this reason:
{error}

Return corrected JSON. Use only values present in the model output above.
"""