import logging
from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    from .routes import register_routes
    from .errors import register_error_handlers

    register_routes(app)
    register_error_handlers(app)

    return app
