"""
User model. [Demo purpose only]
"""

from pydantic import BaseModel, EmailStr

class User(BaseModel):
    username: str
    email: EmailStr
    is_active: bool = True
