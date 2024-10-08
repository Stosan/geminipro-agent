# encoding: utf-8
from pydantic_settings import BaseSettings
from src.config import appconfig



class Settings(BaseSettings):
    """
    This class extends the BaseSettings class from FastAPI.
    It contains the project definitions.

    Args:
        None.

    Returns:
        class: extends the settings class.
    """

    if appconfig.ENV == "development":
        API_STR: str = "/dev/api/v1"
    else:
        API_STR: str = "/api/v1"

    VERSION: str = "3.0.2"
    
    PROJECT_NAME: str = "AI Server"

  

def get_setting():
    """
    Return the settings object.

    Args:
        None.

    Returns:
        class: extends the settings class.
    """
    return Settings()

