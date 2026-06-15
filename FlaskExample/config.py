import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DB_HOST: str = os.environ.get('DB_HOST', 'localhost')
    DB_PORT: int = int(os.environ.get('DB_PORT', '3306'))
    DB_NAME: str = os.environ.get('DB_NAME', 'test')
    DB_USER: str = os.environ.get('DB_USER', 'root')
    DB_PASSWORD: str = os.environ.get('DB_PASSWORD', '')
    DEBUG: bool = os.environ.get('DEBUG', 'false').lower() == 'true'
    HOST: str = os.environ.get('HOST', '0.0.0.0')
    PORT: int = int(os.environ.get('PORT', '5000'))
