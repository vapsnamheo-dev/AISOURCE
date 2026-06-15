from app.models.schemas import BmiResult
from app.repositories.bmi_repository import BmiRepository

bmi_repository = BmiRepository()

_CATEGORIES: list[tuple[float, str]] = [
    (18.5, "저체중"),
    (23.0, "정상"),
    (25.0, "과체중"),
    (30.0, "비만"),
]


def _get_category(bmi: float) -> str:
    for threshold, category in _CATEGORIES:
        if bmi < threshold:
            return category
    return "고도비만"


def calculate_and_save(weight: float, height: float) -> BmiResult:
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)
    category = _get_category(bmi)
    bmi_repository.save(weight, height, bmi, category)
    return BmiResult(bmi=bmi, category=category, weight=weight, height=height)


def get_history(limit: int = 10) -> list[dict]:
    return bmi_repository.find_recent(limit)
