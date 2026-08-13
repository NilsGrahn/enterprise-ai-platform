from dataclasses import dataclass, asdict, field


@dataclass
class FeatureContribution:
    """One feature's contribution to one prediction."""

    feature: str            # technical name, e.g. 'revolving_utilisation'
    display_name: str       # human label, e.g. 'Credit utilisation ratio'
    value: float            # the applicant's actual value for this feature
    shap_value: float       # log-odds contribution (+ pushes risk up)
    direction: str          # 'increases_risk' | 'decreases_risk'
    rank: int               # 1 = largest absolute contribution

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplanationResult:
    """The complete explanation for one prediction.

    This is the contract between the explain service, the LLM service, and
    the API. Nothing downstream reads a raw SHAP array.
    """

    request_id: str
    probability: float
    base_value: float                   # expected log-odds over training data
    contributions: list                 # list[FeatureContribution], top-N by |shap|
    model_version: str
    pipeline_name: str
    all_contributions: list = field(default_factory=list)   # every feature, unranked

    def to_dict(self) -> dict:
        return {
            'request_id': self.request_id,
            'probability': self.probability,
            'base_value': self.base_value,
            'contributions': [c.to_dict() for c in self.contributions],
            'model_version': self.model_version,
            'pipeline_name': self.pipeline_name,
        }

    def increases_risk(self) -> list:
        """Contributions pushing risk up, in rank order."""
        return [c for c in self.contributions if c.direction == 'increases_risk']

    def decreases_risk(self) -> list:
        """Contributions pushing risk down, in rank order."""
        return [c for c in self.contributions if c.direction == 'decreases_risk']