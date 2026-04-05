from pathlib import Path

import joblib

from src.preprocess import preprocess_text
from src.train import train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        train()
    return joblib.load(model_path)


def predict_sentiment(text: str, model=None) -> dict:
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    model = model or load_model()
    cleaned_text = preprocess_text(text)
    prediction = int(model.predict([cleaned_text])[0])
    probabilities = model.predict_proba([cleaned_text])[0]

    label = "positive" if prediction == 1 else "negative"
    confidence = float(max(probabilities))
    return {
        "text": text,
        "processed_text": cleaned_text,
        "label": label,
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    sample = "I love how smooth this experience is."
    print(predict_sentiment(sample))
