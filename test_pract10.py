import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pract10 import (
    build_user_recommendations,
    compute_metrics,
    load_spotify_as_reviews,
    mask_name,
    mask_phone,
    predict_sentiment,
)


def test_mask_phone():
    assert mask_phone("+79991234567") == "+799***4567"


def test_mask_name():
    assert mask_name("Алина") == "Ал*на"


def test_predict_sentiment_output():
    texts = np.array(["хороший трек", "плохой трек", "отличный звук", "ужасный звук"])
    y = np.array([1, 0, 1, 0])
    vec = TfidfVectorizer()
    x = vec.fit_transform(texts)
    model = LogisticRegression(max_iter=1000).fit(x, y)
    label, prob = predict_sentiment("хороший звук", vec, model)
    assert label in {"позитивный", "негативный"}
    assert 0 <= prob <= 100


def test_load_data_schema_and_size():
    df = load_spotify_as_reviews("spotify-tracks-dataset-detailed.csv", sample_size=250, random_state=1)
    assert len(df) == 250
    expected = {"user_id", "product", "text", "rating", "review_date", "sentiment"}
    assert expected.issubset(df.columns)


def test_compute_metrics_keys():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.4, 0.1])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert {"Accuracy", "Precision", "Recall", "F1", "ROC_AUC"}.issubset(metrics.keys())


def test_recommendations_not_crash():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "product": ["rock", "pop", "rock", "jazz", "pop", "jazz"],
            "rating": [5, 2, 4, 5, 5, 4],
        }
    )
    recs = build_user_recommendations(df, "u1", top_n=3)
    assert isinstance(recs, list)
