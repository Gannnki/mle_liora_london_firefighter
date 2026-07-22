"""FastAPI entry point for London Fire Brigade response-time inference."""

from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference_service import ArtifactError, PredictionService


class PredictionRequest(BaseModel):
    template: dict[str, Any]
    month: int = Field(ge=1, le=12)
    weekday: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)
    incident_group: Literal["Fire", "Special Service"]
    special_service_type: str = Field(min_length=1)
    property_category: str = Field(min_length=1)
    property_type: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    attendance_time_seconds: float
    minutes: int
    remaining_seconds: int
    inference_ms: float


app = FastAPI(
    title="LFB Response Time API",
    version="1.0.0",
    description="Inference API used by the Streamlit live simulator.",
)


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService()


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_prediction_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return {"status": "ok", "model": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, float | int]:
    try:
        service = get_prediction_service()
        settings = request.model_dump(exclude={"template"})
        return service.predict(request.template, settings)
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid prediction input: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
