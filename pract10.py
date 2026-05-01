import logging
import sqlite3
import uuid
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from xgboost import XGBClassifier

DB_PATH = "reviews.db"
DATASET_PATH = "spotify-tracks-dataset-detailed.csv"
APP_LOG_PATH = "app.log"

logging.basicConfig(
    filename=APP_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def log_action(user_id: str, action: str, result: str):
    logging.info("user_id=%s | action=%s | result=%s", user_id, action, result)


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product TEXT NOT NULL,
            text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            review_date TEXT NOT NULL,
            sentiment INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def mask_phone(phone: str) -> str:
    if not phone:
        return ""
    return phone[:4] + "***" + phone[-4:] if len(phone) >= 10 else phone


def mask_name(name: str) -> str:
    if not name:
        return ""
    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)
    return name[:2] + "*" * (len(name) - 4) + name[-2:]


def load_spotify_as_reviews(path: str, sample_size: int = 2200, random_state: int = 42) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.dropna(subset=["track_name", "track_genre", "popularity"]).copy()
    if len(raw) > sample_size:
        raw = raw.sample(sample_size, random_state=random_state).copy()
    raw = raw.reset_index(drop=True)

    popularity = raw["popularity"].fillna(0).astype(float)
    ratings = pd.qcut(popularity.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    energy = np.where(raw["energy"].fillna(0.5) > 0.6, "энергичный", "спокойный")
    explicit = np.where(raw["explicit"].astype(str).str.lower() == "true", "explicit", "clean")
    reviews = (
        "трек "
        + raw["track_name"].astype(str)
        + ", жанр "
        + raw["track_genre"].astype(str)
        + ", "
        + energy
        + ", "
        + explicit
    )

    now = datetime.now()
    review_dates = [
        (now - timedelta(days=int(v))).strftime("%Y-%m-%d")
        for v in np.random.randint(0, 120, size=len(raw))
    ]

    df = pd.DataFrame(
        {
            "user_id": [f"seed_user_{i}" for i in np.random.randint(1, 200, size=len(raw))],
            "product": raw["track_genre"].astype(str),
            "text": reviews,
            "rating": ratings,
            "review_date": review_dates,
        }
    )
    df["sentiment"] = np.where(df["rating"] >= 4, 1, np.where(df["rating"] <= 2, 0, np.nan))
    return df


def seed_initial_data():
    conn = get_connection()
    existing = pd.read_sql_query("SELECT COUNT(*) AS c FROM reviews", conn)["c"].iloc[0]
    if existing < 200:
        seed = load_spotify_as_reviews(DATASET_PATH, sample_size=2200)
        seed.to_sql("reviews", conn, if_exists="append", index=False)
    conn.close()


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    conn.close()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df


def add_review(user_id: str, product: str, text: str, rating: int, sentiment: int):
    conn = get_connection()
    conn.execute(
        "INSERT INTO reviews(user_id, product, text, rating, review_date, sentiment) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, product, text, int(rating), datetime.now().strftime("%Y-%m-%d"), int(sentiment)),
    )
    conn.commit()
    conn.close()
    st.cache_data.clear()


@st.cache_resource(show_spinner=False)
def build_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(list(texts), show_progress_bar=False)
    return np.array(emb)


def compute_metrics(y_true, y_pred, y_proba=None):
    data = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        data["ROC_AUC"] = roc_auc_score(y_true, y_proba)
    return data


@st.cache_resource(show_spinner=False)
def train_and_tune_models(df_binary: pd.DataFrame):
    x_text = df_binary["text"].astype(str).values
    y = df_binary["sentiment"].astype(int).values
    vectorizer = TfidfVectorizer(max_features=700, ngram_range=(1, 2))
    x = vectorizer.fit_transform(x_text)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1500, random_state=42),
            {"C": [0.3, 1.0, 3.0], "solver": ["liblinear", "lbfgs"], "class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [120, 200], "max_depth": [None, 12, 20], "min_samples_split": [2, 5]},
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {"n_estimators": [120, 220], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
        ),
        "MLPClassifier": (
            MLPClassifier(max_iter=400, random_state=42),
            {"hidden_layer_sizes": [(64,), (128,)], "alpha": [0.0001, 0.001], "learning_rate_init": [0.001, 0.01]},
        ),
    }
    if CatBoostClassifier is not None:
        models["CatBoost"] = (
            CatBoostClassifier(verbose=0, random_state=42),
            {"depth": [4, 6], "learning_rate": [0.03, 0.1], "iterations": [120, 200]},
        )
    else:
        pass

    before_rows = []
    after_rows = []
    roc_data = {}
    best_models = {}

    for name, (model, grid) in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        m_before = compute_metrics(y_test, pred, prob)
        m_before["Model"] = name
        m_before["Stage"] = "Before"
        before_rows.append(m_before)

        gs = GridSearchCV(model, grid, scoring="f1", cv=3, n_jobs=-1)
        gs.fit(x_train, y_train)
        tuned = gs.best_estimator_
        pred_tuned = tuned.predict(x_test)
        prob_tuned = tuned.predict_proba(x_test)[:, 1] if hasattr(tuned, "predict_proba") else None
        m_after = compute_metrics(y_test, pred_tuned, prob_tuned)
        m_after["Model"] = name
        m_after["Stage"] = "After"
        after_rows.append(m_after)
        best_models[name] = tuned

        if prob_tuned is not None:
            fpr, tpr, _ = roc_curve(y_test, prob_tuned)
            roc_data[name] = {"fpr": fpr, "tpr": tpr}

    metrics_df = pd.DataFrame(before_rows + after_rows)
    metrics_pivot = metrics_df.pivot_table(index="Model", columns="Stage", values=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
    return vectorizer, best_models, metrics_df, metrics_pivot, roc_data


@st.cache_resource(show_spinner=False)
def train_quick_model(df_binary: pd.DataFrame):
    x_text = df_binary["text"].astype(str).values
    y = df_binary["sentiment"].astype(int).values
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    x = vectorizer.fit_transform(x_text)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x, y)
    return vectorizer, model


def predict_sentiment(text: str, vectorizer: TfidfVectorizer, model):
    xv = vectorizer.transform([text])
    pred = int(model.predict(xv)[0])
    prob = float(model.predict_proba(xv)[0][pred]) if hasattr(model, "predict_proba") else 0.5
    return ("позитивный" if pred == 1 else "негативный", prob * 100)


def build_user_recommendations(df: pd.DataFrame, target_user: str, top_n: int = 5):
    matrix = df.pivot_table(index="user_id", columns="product", values="rating", aggfunc="mean")
    if target_user not in matrix.index:
        return []
    filled = matrix.fillna(0)
    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=filled.index, columns=filled.index)
    neighbors = sim_df[target_user].sort_values(ascending=False).iloc[1:11]

    seen = set(df[df["user_id"] == target_user]["product"])
    scores = {}
    for user, w in neighbors.items():
        if w <= 0:
            continue
        for prod, rate in matrix.loc[user].dropna().items():
            if prod in seen:
                continue
            scores.setdefault(prod, {"num": 0.0, "den": 0.0})
            scores[prod]["num"] += float(rate) * float(w)
            scores[prod]["den"] += float(w)
    out = [(prod, v["num"] / v["den"]) for prod, v in scores.items() if v["den"] > 0]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


def get_top_negative_words(df_binary: pd.DataFrame, top_n: int = 10):
    txt = " ".join(df_binary[df_binary["sentiment"] == 0]["text"].astype(str).str.lower().tolist())
    words = [w for w in txt.replace(",", " ").split() if len(w) > 2]
    stop = {"трек", "жанр", "clean", "explicit", "the", "and"}
    return Counter([w for w in words if w not in stop]).most_common(top_n)


def product_stats(df: pd.DataFrame):
    tmp = df.copy()
    tmp["positive"] = (tmp["rating"] >= 4).astype(int)
    return (
        tmp.groupby("product")
        .agg(avg_rating=("rating", "mean"), reviews_count=("rating", "count"), positive_share=("positive", "mean"))
        .reset_index()
        .sort_values(["avg_rating", "reviews_count"], ascending=[False, False])
    )


def cluster_reviews(df_binary: pd.DataFrame):
    texts = tuple(df_binary["text"].astype(str).tolist())
    embeddings = build_embeddings(texts)

    km = KMeans(n_clusters=6, random_state=42, n_init="auto")
    km_labels = km.fit_predict(embeddings)
    db = DBSCAN(eps=1.2, min_samples=8)
    db_labels = db.fit_predict(embeddings)

    reducer = PCA(n_components=2, random_state=42)
    points = reducer.fit_transform(embeddings)

    out = df_binary.copy().reset_index(drop=True)
    out["x"] = points[:, 0]
    out["y"] = points[:, 1]
    out["cluster_kmeans"] = km_labels
    out["cluster_dbscan"] = db_labels
    return out


def cluster_insights(cluster_df: pd.DataFrame, cluster_col: str):
    rows = []
    for cid in sorted(cluster_df[cluster_col].dropna().unique()):
        part = cluster_df[cluster_df[cluster_col] == cid]
        text = " ".join(part["text"].astype(str).str.lower().tolist())
        words = [w for w in text.replace(",", " ").split() if len(w) > 2]
        top_words = ", ".join([w for w, _ in Counter(words).most_common(10)])
        rows.append(
            {
                "cluster": int(cid),
                "size": len(part),
                "avg_positive_share": float((part["sentiment"] == 1).mean()),
                "top_words": top_words,
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def ensure_user_session():
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = f"user_{uuid.uuid4().hex[:8]}"
    return st.session_state["user_id"]


def show_home_page(df, user_id, vectorizer, best_model):
    st.title("ML Product - уровень 5")
    st.caption(f"user_id: {user_id}")
    log_action(user_id, "page_view", "home")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Новый отзыв")
        product = st.selectbox("Товар/жанр", sorted(df["product"].dropna().unique().tolist()))
        rating = st.slider("Оценка", 1, 5, 4)
        text = st.text_area("Текст", "Мне понравился этот трек, крутая энергия")
        if st.button("Сохранить отзыв"):
            label, conf = predict_sentiment(text, vectorizer, best_model)
            sentiment_num = 1 if label == "позитивный" else 0
            add_review(user_id, product, text, rating, sentiment_num)
            log_action(user_id, "add_review", f"ok:{label}:{conf:.1f}")
            st.success(f"Отзыв добавлен. Тональность: {label} ({conf:.1f}%)")
    with col2:
        st.subheader("Персональные рекомендации")
        recs = build_user_recommendations(df, user_id)
        if recs:
            st.dataframe(pd.DataFrame(recs, columns=["product", "pred_rating"]), use_container_width=True)
        else:
            st.info("Нужно больше истории оценок для рекомендаций.")

    st.subheader("Базовая аналитика")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x="rating", nbins=5, title="Распределение оценок"), use_container_width=True)
    with c2:
        daily = df.groupby(df["review_date"].dt.date).size().reset_index(name="count")
        daily["review_date"] = pd.to_datetime(daily["review_date"])
        st.plotly_chart(px.line(daily, x="review_date", y="count", title="Динамика отзывов"), use_container_width=True)

    df_bin = df[df["sentiment"].isin([0, 1])].copy()
    words = get_top_negative_words(df_bin, top_n=10)
    if words:
        wd = pd.DataFrame(words, columns=["word", "count"])
        st.plotly_chart(px.bar(wd, x="word", y="count", title="Облако слов (top negative words)"), use_container_width=True)

    daily2 = df.groupby(df["review_date"].dt.date).size().reset_index(name="reviews_count")
    daily2["review_date"] = pd.to_datetime(daily2["review_date"])
    daily2 = daily2.sort_values("review_date")
    daily2["dow"] = daily2["review_date"].dt.dayofweek
    for lag in [1, 2, 3]:
        daily2[f"lag_{lag}"] = daily2["reviews_count"].shift(lag)
    daily2 = daily2.dropna()
    if len(daily2) >= 10:
        cols = ["dow", "lag_1", "lag_2", "lag_3"]
        xtr = daily2[cols].values[:-3]
        ytr = daily2["reviews_count"].values[:-3]
        xte = daily2[cols].values[-3:]
        yte = daily2["reviews_count"].values[-3:]
        reg = LinearRegression()
        reg.fit(xtr, ytr)
        pred = reg.predict(xte)
        forecast_df = pd.DataFrame({"day": [1, 2, 3], "actual": yte, "pred": pred})
        st.plotly_chart(px.bar(forecast_df, x="day", y=["actual", "pred"], barmode="group", title="Прогноз отзывов"), use_container_width=True)


def show_models_page(metrics_df, metrics_pivot, roc_data, user_id):
    st.title("Сравнение моделей")
    log_action(user_id, "page_view", "models")
    st.dataframe(metrics_df.round(3), use_container_width=True)
    st.subheader("До/после GridSearchCV")
    st.dataframe(metrics_pivot.round(3), use_container_width=True)

    roc_plot_df = []
    for name, vals in roc_data.items():
        roc_plot_df.append(pd.DataFrame({"fpr": vals["fpr"], "tpr": vals["tpr"], "model": name}))
    if roc_plot_df:
        roc_all = pd.concat(roc_plot_df, ignore_index=True)
        fig = px.line(roc_all, x="fpr", y="tpr", color="model", title="ROC curves")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)


def show_clusters_page(cluster_df, user_id):
    st.title("Кластеризация отзывов")
    log_action(user_id, "page_view", "clusters")
    algo = st.radio("Алгоритм", ["KMeans", "DBSCAN"], horizontal=True)
    cluster_col = "cluster_kmeans" if algo == "KMeans" else "cluster_dbscan"

    fig = px.scatter(
        cluster_df,
        x="x",
        y="y",
        color=cluster_col,
        hover_data=["text", "product", "rating"],
        title=f"Кластеры ({algo})",
    )
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    st.subheader("Инсайты по кластерам")
    st.dataframe(cluster_insights(cluster_df, cluster_col), use_container_width=True)

    selected_cluster = None
    if hasattr(event, "selection") and event.selection and event.selection.get("points"):
        try:
            selected_cluster = event.selection["points"][0]["customdata"][0]
        except Exception:
            selected_cluster = None

    if selected_cluster is None:
        selected_cluster = st.selectbox("Выберите кластер", sorted(cluster_df[cluster_col].dropna().unique().tolist()))
    examples = cluster_df[cluster_df[cluster_col] == selected_cluster][["text", "product", "rating"]].head(10)
    st.write(f"Примеры из кластера: {selected_cluster}")
    st.dataframe(examples, use_container_width=True)


def show_products_page(df, user_id):
    st.title("Статистика по товарам")
    log_action(user_id, "page_view", "products")
    stats = product_stats(df)
    st.dataframe(stats.round(3), use_container_width=True)


def show_logs_page(user_id):
    st.title("Логи приложения (админ)")
    pwd = st.text_input("Admin password", type="password")
    if pwd != "admin123":
        st.warning("Только для разработчика/администратора")
        log_action(user_id, "view_logs", "denied")
        return
    log_action(user_id, "view_logs", "allowed")
    if not Path(APP_LOG_PATH).exists():
        st.info("Лог-файл пока пуст.")
        return
    with open(APP_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-50:]
    st.code("".join(lines), language="text")


def main():
    st.set_page_config(page_title="Pract10 Grade 5", layout="wide")
    init_db()
    seed_initial_data()
    user_id = ensure_user_session()
    df = load_reviews()
    df_binary = df[df["sentiment"].isin([0, 1])].copy()

    page = st.sidebar.radio("Раздел", ["Главная", "Сравнение моделей", "Кластеры", "Товары", "Логи"])
    if page == "Главная":
        vectorizer, best_model = train_quick_model(df_binary)
        show_home_page(df, user_id, vectorizer, best_model)
    elif page == "Сравнение моделей":
        with st.spinner("Обучаем 5 моделей и подбираем гиперпараметры..."):
            vectorizer, best_models, metrics_df, metrics_pivot, roc_data = train_and_tune_models(df_binary)
        _ = vectorizer
        _ = best_models
        show_models_page(metrics_df, metrics_pivot, roc_data, user_id)
    elif page == "Кластеры":
        with st.spinner("Строим эмбеддинги и кластеризацию..."):
            try:
                cluster_df = cluster_reviews(df_binary.sample(min(1200, len(df_binary)), random_state=42))
                show_clusters_page(cluster_df, user_id)
            except ModuleNotFoundError as e:
                st.error(f"Не хватает зависимости: {e}")
                st.info("Установите пакет в активном .venv: pip install sentence-transformers")
    elif page == "Товары":
        show_products_page(df, user_id)
    else:
        show_logs_page(user_id)


if __name__ == "__main__":
    main()
