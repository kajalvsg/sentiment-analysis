import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources() -> None:
    """Download NLTK resources needed for preprocessing."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_resources()
_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """Clean and normalize raw text into lemmas."""
    cleaned = _clean_text(text)
    tokens = cleaned.split()
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    lemmas = [_LEMMATIZER.lemmatize(token) for token in tokens]
    return " ".join(lemmas)


def preprocess_corpus(texts: Iterable[str]) -> List[str]:
    return [preprocess_text(text) for text in texts]
