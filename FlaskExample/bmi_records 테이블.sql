USE test;  

-- 테이블 생성
CREATE TABLE IF NOT EXISTS bmi_records (

    id INT AUTO_INCREMENT PRIMARY KEY,
    weight FLOAT NOT NULL,
    height FLOAT NOT NULL,
    bmi FLOAT NOT NULL,
    category VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);