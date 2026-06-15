from dataclasses import dataclass


@dataclass(frozen=True)
class BmiResult:
    bmi: float
    category: str
    weight: float
    height: float
