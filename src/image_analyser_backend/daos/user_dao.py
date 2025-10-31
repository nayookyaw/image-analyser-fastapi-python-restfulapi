"""
Data Access Object for User model. [Demo purpose only]
"""

from fastapi import Depends
# from sqlalchemy import select
# from sqlalchemy.ext.asyncio import AsyncSession
# from app.api.deps import get_db
from image_analyser_backend.models.users import User

class UserDao:
    @classmethod
    async def get_user_by_email(cls, email: str) -> User | None:
        pass
        # _db: AsyncSession = Depends(get_db)
        # exist_user = await _db.execute(select(User).where(User.email == email))
        # return exist_user.scalar_one_or_none()