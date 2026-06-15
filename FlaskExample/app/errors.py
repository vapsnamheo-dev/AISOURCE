import logging
from flask import Flask, render_template

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def handle_api_error(e: ApiError):
        return render_template('index.html', error=e.message), e.status_code

    @app.errorhandler(500)
    def handle_500(e: Exception):
        logger.exception("서버 오류 발생")
        return render_template('index.html', error="서버 오류가 발생했습니다."), 500
