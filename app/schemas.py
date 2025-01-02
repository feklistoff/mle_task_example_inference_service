from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    venue_id: str = Field(examples=["8a61c05"])
    time_received: datetime = Field(examples=["2024-12-02T09:50:01.897036"])
    is_retail: int = Field(examples=[1])

    @field_validator("is_retail")
    def validate_is_retail(cls, value):
        if value not in (0, 1):
            raise ValueError("is_retail must be either 0 or 1")
        return value


class PredictionResponse(BaseModel):
    delivery_duration: float
