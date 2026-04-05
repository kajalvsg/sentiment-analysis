from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocess import preprocess_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "twitter_sentiment.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def build_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1500, random_state=42),
        "NaiveBayes": MultinomialNB(),
    }


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    if "tweet" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'tweet' and 'label' columns.")

    df = df[["tweet", "label"]].dropna()
    df["processed_text"] = preprocess_corpus(df["tweet"])

    x_train, x_test, y_train, y_test = train_test_split(
        df["processed_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    best_model_name = ""
    best_f1 = -1.0
    best_pipeline = None

    for model_name, classifier in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        results.append({"model": model_name, **metrics})

        model_path = MODEL_DIR / f"{model_name.lower()}_pipeline.joblib"
        joblib.dump(pipeline, model_path)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("Model training failed to produce a best pipeline.")

    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    results_df.to_csv(OUTPUTS_DIR / "model_results.csv", index=False)

    print("Training complete.")
    print(f"Best model: {best_model_name} (F1={best_f1:.4f})")
    print(f"Saved model artifacts to: {MODEL_DIR}")
    print(f"Saved evaluation metrics to: {OUTPUTS_DIR / 'model_results.csv'}")


if __name__ == "__main__":
    train()
