from typing import Generic, TypeVar, Optional
from pydantic import BaseModel
from starlette.status import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR

T = TypeVar("T")

class Response_SUCCESS(BaseModel, Generic[T]):
    status_code: int = HTTP_200_OK
    message: Optional[str] = None
    data: Optional[T] = None

class Response_ERROR(BaseModel):
    status_code: int = HTTP_500_INTERNAL_SERVER_ERROR
    message: Optional[str] = None