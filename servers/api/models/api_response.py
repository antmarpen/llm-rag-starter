from typing import Optional, TypeVar, Generic

from pydantic import BaseModel


class APIResponse(BaseModel):
    success: bool = False
    data: Optional[str] = None


T = TypeVar("T")
class APIResponseWithData(APIResponse, Generic[T]):
    data: Optional[T] = None