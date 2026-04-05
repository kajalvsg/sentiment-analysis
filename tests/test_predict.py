from src.predict import predict_sentiment


class DummyModel:
    def predict(self, features):
        _ = features
        return [1]

    def predict_proba(self, features):
        _ = features
        return [[0.1, 0.9]]


def test_predict_sentiment_with_mock_model():
    result = predict_sentiment("This is excellent", model=DummyModel())
    assert result["label"] == "positive"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["processed_text"] != ""
