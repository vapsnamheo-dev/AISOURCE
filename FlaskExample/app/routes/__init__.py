from flask import Flask
from .bmi import bmi_bp


def register_routes(app: Flask) -> None:
    app.register_blueprint(bmi_bp)
