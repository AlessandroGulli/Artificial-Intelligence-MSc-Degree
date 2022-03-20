
"""Flask config."""
from os import environ, path
from dotenv import load_dotenv

BASE_DIR = path.abspath(path.dirname(__file__))
load_dotenv(path.join(BASE_DIR, '.env'))


class Config:
    """Flask configuration variables."""

    # General Config
    FLASK_APP = environ.get('FLASK_APP')
    FLASK_ENV = environ.get('FLASK_ENV')
    SECRET_KEY = environ.get('SECRET_KEY')
    # Static Assets
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'
    COMPRESSOR_DEBUG = environ.get('COMPRESSOR_DEBUG')
    # Database
    SQLALCHEMY_DATABASE_URI = "sqlite:///watering_system.sqlite"  
    SQLALCHEMY_BINDS = {"plants_info":"sqlite:///plants_info.sqlite"}
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Flask-Caching related configs
    CACHE_TYPE = "null"  
    SEND_FILE_MAX_AGE_DEFAULT = 0
