from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_sentiment


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify sentiment")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    processed_text: str


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    app_instance.state.model = load_model()
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="TF-IDF based sentiment classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predict_sentiment(payload.text, model=app.state.model)
        return PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
            processed_text=result["processed_text"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
