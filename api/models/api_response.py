from pydantic import BaseModel
from typing import Any, Optional, TypeVar, Generic


class APIResponse(BaseModel):
    success: bool = False
    message: Optional[str] = None


T = TypeVar("T")
class APIResponseWithData(APIResponse, Generic[T]):
    data: Optional[T] = None