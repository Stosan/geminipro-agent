# Instantiate basicAuth
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.config import appconfig


security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    """
    This function sets up the basic auth url protection and returns the credential name.

    Args:
        credentials (HTTPBasicCredentials): Basic auth credentials.

    Raises:
        HTTPException: If the username or password is incorrect.

    Returns:
        str: The username from the credentials.
    """
    correct_username = secrets.compare_digest(credentials.username, appconfig.AUTH_USER)
    correct_password = secrets.compare_digest(
        credentials.password, appconfig.AUTH_PASSWORD
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect userid or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username