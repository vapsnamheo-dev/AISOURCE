# 사전 설치 : pip install flask pymysql requests
from flask import Flask, render_template, request, redirect, url_for
from bmi import BMICalculator
from db import Database
from config import Config
import atexit   # 애플리케이션 종료시 실행을 요청 (ex. DB연결 종료)

app = Flask(__name__)   # Flask 앱 초기화
db = Database()   # DB 초기화

# 애플리케이션 종료 시 DB 연결 종료
atexit.register(db.close)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        
        # 입력값 유효성 검사
        if weight <= 0 or height <= 0:
            return render_template('index.html', error="체중과 신장은 양수여야 합니다.")
        
        # BMI 계산
        calculator = BMICalculator(weight, height)
        result = calculator.get_result()
        
        # 데이터베이스에 저장
        db.save_bmi_record(weight, height, result["bmi"], result["category"])
        
        return render_template('result.html', 
                              bmi=result["bmi"], 
                              category=result["category"],
                              weight=weight,
                              height=height)
    except ValueError:
        return render_template('index.html', error="유효한 숫자를 입력해주세요.")

@app.route('/history')
def history():
    # 최근 BMI 기록 10개 가져오기
    records = db.get_bmi_records(10)
    return render_template('history.html', records=records)

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)