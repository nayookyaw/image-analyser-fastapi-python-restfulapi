from pydantic import BaseModel, Field
from typing import Annotated

class ConfigUpdateRequestBody(BaseModel):
    parameter_a: bool = Field(..., description="Parameter A is required")
    parameter_b: Annotated[
        float,Field(..., ge=0.0,le=1.0,description="Parameter B is required")
    ]
