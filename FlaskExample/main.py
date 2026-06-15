from app import create_app
from config import Config

flask_app = create_app()

if __name__ == '__main__':
    flask_app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
