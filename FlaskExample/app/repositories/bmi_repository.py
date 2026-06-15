import atexit
import logging

import pymysql
from pymysql import Error

from config import Config

logger = logging.getLogger(__name__)


class BmiRepository:
    def __init__(self) -> None:
        self._connection: pymysql.Connection | None = None
        self._connect()
        atexit.register(self.close)

    def _connect(self) -> None:
        try:
            self._connection = pymysql.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
            )
            logger.info("MariaDB에 성공적으로 연결되었습니다.")
        except Error as e:
            logger.error("MariaDB 연결 중 오류 발생: %s", e)

    def save(self, weight: float, height: float, bmi: float, category: str) -> bool:
        if self._connection is None:
            logger.warning("데이터베이스 연결이 없습니다.")
            return False
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO bmi_records (weight, height, bmi, category) VALUES (%s, %s, %s, %s)",
                    (weight, height, bmi, category),
                )
            self._connection.commit()
            logger.info("BMI 기록이 저장되었습니다.")
            return True
        except Error as e:
            logger.error("데이터 저장 중 오류 발생: %s", e)
            return False

    def find_recent(self, limit: int = 10) -> list[dict]:
        if self._connection is None:
            logger.warning("데이터베이스 연결이 없습니다.")
            return []
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM bmi_records ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                return cursor.fetchall()
        except Error as e:
            logger.error("데이터 조회 중 오류 발생: %s", e)
            return []

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            logger.info("MariaDB 연결이 종료되었습니다.")
