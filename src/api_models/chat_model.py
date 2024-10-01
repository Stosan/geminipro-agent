
from typing import Optional
from pydantic import BaseModel

class UserData(BaseModel):
    name: str = ""
    gender: str = ""
    current_location: str = ""
    timezone: str = ""

class ChatRequest(BaseModel):
    """Request model for chat requests.
    the message from the user.
    """
    sentence: str
    userData: Optional[UserData] = None
