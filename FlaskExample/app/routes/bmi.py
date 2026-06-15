from flask import Blueprint, render_template, request

from app.errors import ApiError
from app.services import bmi_service

bmi_bp = Blueprint('bmi', __name__)


@bmi_bp.get('/')
def index():
    return render_template('index.html')


@bmi_bp.post('/calculate')
def calculate():
    try:
        weight = float(request.form['weight'])
        height = float(request.form['height'])
    except (ValueError, KeyError):
        raise ApiError("유효한 숫자를 입력해주세요.")

    if weight <= 0 or height <= 0:
        raise ApiError("체중과 신장은 양수여야 합니다.")

    result = bmi_service.calculate_and_save(weight, height)
    return render_template('result.html', result=result)


@bmi_bp.get('/history')
def history():
    records = bmi_service.get_history(10)
    return render_template('history.html', records=records)
