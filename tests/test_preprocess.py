from src.preprocess import preprocess_text


def test_preprocess_text_removes_noise():
    raw = "I LOVE this product!!! Visit https://example.com #Amazing @user"
    cleaned = preprocess_text(raw)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "love" in cleaned
