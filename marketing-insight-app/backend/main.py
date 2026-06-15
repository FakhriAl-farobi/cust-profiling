from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_FALLBACK = ROOT_DIR / "dataset" / "market_analyst_12000_transactions.csv"
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:254981@localhost:5432/marketing_insight")

engine = create_engine(DB_URL, pool_pre_ping=True) if DB_URL else None

app = FastAPI(title="Artajasa Customer Profiling API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_mcc_ranges() -> dict[str, tuple[int, int]]:
    return {
        "Agricultural Services": (1, 1499),
        "Contracted Services": (1500, 2999),
        "Airlines": (3000, 3299),
        "Car Rental": (3300, 3499),
        "Lodging (Hotels)": (3500, 3999),
        "Transportation Services": (4000, 4799),
        "Utility Services": (4800, 4999),
        "Retail Outlet Services": (5000, 5599),
        "Clothing Stores": (5600, 5699),
        "Miscellaneous Stores": (5700, 7299),
        "Business Services": (7300, 7999),
        "Professional Services & Orgs": (8000, 8999),
        "Government Services": (9000, 9999),
    }


CATEGORY_TO_MCC = {
    "F&B": 5812,
    "Retail": 5411,
    "Transport": 4121,
    "Healthcare": 8099,
    "Education": 8299,
}


class FilterPayload(BaseModel):
    industries: list[str] = Field(default_factory=lambda: ["Retail Outlet Services", "Miscellaneous Stores"])
    cpan_limit: int = 5000
    row_limit: int = 200000
    start_date: str | None = None
    end_date: str | None = None
    acquirers: list[str] = Field(default_factory=list)
    issuers: list[str] = Field(default_factory=list)
    merchant_types: list[str] = Field(default_factory=list)


class ClusterPayload(FilterPayload):
    rfm_segments: list[str] = Field(default_factory=lambda: ["Hibernating", "At Risk", "About To Sleep"])
    features: list[str] = Field(default_factory=lambda: ["trx_count", "total_amount", "recency_days", "avg_hour_sin", "avg_hour_cos"])
    model_type: Literal["KMeans", "MiniBatch KMeans", "Gaussian Mixture", "DBSCAN", "HDBSCAN"] = "MiniBatch KMeans"
    n_clusters: int = 4
    batch_size: int = 1024
    eps: float = 0.5
    min_samples: int = 8
    min_cluster_size: int = 20


class ChatPayload(BaseModel):
    message: str
    profile: dict[str, Any] | None = None


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "nominal": "amount",
        "nama_merchant": "merchant_name",
        "category": "mcc_description",
        "lokasi_merchant": "merchant_location",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    if "timestamp" not in df.columns:
        raise HTTPException(422, "Dataset must contain a timestamp column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)

    if "customer_id_masked" not in df.columns:
        base = df.get("issuer", "QRIS").astype(str) + "-" + df.get("merchant_name", "MERCHANT").astype(str)
        df["customer_id_masked"] = ["CPAN-" + str(abs(hash(v + str(i % 173))) % 900000) for i, v in enumerate(base)]
    if "merchant_type" not in df.columns:
        df["merchant_type"] = df.get("mcc_description", "Retail").map(CATEGORY_TO_MCC).fillna(5999).astype(int)
    if "mcc_description" not in df.columns:
        df["mcc_description"] = df["merchant_type"].astype(str)
    for col in ["acquirer", "issuer", "merchant_name"]:
        if col not in df.columns:
            df[col] = "Unknown"
    return df.dropna(subset=["timestamp"])


def fetch_from_db(industries: list[str], row_limit: int) -> pd.DataFrame:
    if engine is None:
        raise RuntimeError("Database engine is not configured.")
    ranges = get_mcc_ranges()
    conditions = []
    for label in industries:
        start, end = ranges[label]
        conditions.append(f"(CAST(merchant_type AS INTEGER) BETWEEN {start} AND {end})")
    if not conditions:
        return pd.DataFrame()

    query = text(f"""
        SELECT acquirer, issuer, terminal_id, customer_id_masked, processing_code,
               amount, merchant_pan, merchant_type, merchant_crit, merchant_name,
               status_data, timestamp, mcc_description
        FROM data_transaksi_qris
        WHERE {" OR ".join(conditions)}
        LIMIT :row_limit
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"row_limit": row_limit})


def fetch_fallback() -> pd.DataFrame:
    if not CSV_FALLBACK.exists():
        raise HTTPException(503, "Database is unavailable and CSV fallback was not found.")
    return pd.read_csv(CSV_FALLBACK)


def load_data(payload: FilterPayload) -> tuple[pd.DataFrame, str]:
    source = "database"
    try:
        raw = fetch_from_db(payload.industries, payload.row_limit)
    except Exception:
        raw = fetch_fallback()
        source = "csv fallback"

    df = normalize_frame(raw)
    if source == "csv fallback":
        wanted = set(payload.industries)
        if wanted:
            ranges = get_mcc_ranges()
            masks = []
            for label in wanted:
                start, end = ranges.get(label, (0, 9999))
                masks.append(df["merchant_type"].between(start, end))
            if masks:
                mask = masks[0]
                for extra in masks[1:]:
                    mask = mask | extra
                filtered = df[mask]
                df = filtered if not filtered.empty else df

    unique_ids = df["customer_id_masked"].dropna().unique()
    sample_size = min(payload.cpan_limit, len(unique_ids))
    if sample_size:
        sampled = np.random.RandomState(42).choice(unique_ids, size=sample_size, replace=False)
        df = df[df["customer_id_masked"].isin(sampled)]

    if payload.start_date:
        df = df[df["timestamp"].dt.date >= pd.to_datetime(payload.start_date).date()]
    if payload.end_date:
        df = df[df["timestamp"].dt.date <= pd.to_datetime(payload.end_date).date()]
    if payload.acquirers:
        df = df[df["acquirer"].isin(payload.acquirers)]
    if payload.issuers:
        df = df[df["issuer"].isin(payload.issuers)]
    if payload.merchant_types:
        df = df[df["mcc_description"].isin(payload.merchant_types)]
    return df.copy(), source


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = df["timestamp"].max()
    rfm = df.groupby("customer_id_masked").agg(
        Recency=("timestamp", lambda x: (ref_date - x.max()).days),
        Frequency=("timestamp", "count"),
        Monetary=("amount", "sum"),
    ).reset_index()

    def safe_qcut(series: pd.Series, q: int, ascending: bool = True) -> pd.Series:
        try:
            ranks = series.rank(method="first", ascending=ascending)
            return pd.qcut(ranks, q, labels=False, duplicates="drop") + 1
        except Exception:
            return pd.Series([1] * len(series), index=series.index)

    rfm["R-Score"] = safe_qcut(rfm["Recency"], 5, ascending=False)
    rfm["F-Score"] = safe_qcut(rfm["Frequency"], 5)
    rfm["M-Score"] = safe_qcut(rfm["Monetary"], 5)
    rfm["RFM-Score"] = rfm["R-Score"] + rfm["F-Score"] + rfm["M-Score"]
    rfm["FM_Avg"] = ((rfm["F-Score"] + rfm["M-Score"]) / 2).round(1)

    def segment(row: pd.Series) -> str:
        r, fm = row["R-Score"], row["FM_Avg"]
        if r == 5 and fm >= 4: return "Champions"
        if 3 <= r <= 4 and fm >= 4: return "Loyal Customers"
        if r >= 4 and 2 <= fm <= 3.5: return "Potential Loyalist"
        if r == 5 and fm < 2: return "New Customers"
        if r == 4 and fm < 2: return "Promising"
        if r == 3 and 2.5 < fm <= 3.5: return "Needs Attention"
        if r == 3 and fm <= 2.5: return "About To Sleep"
        if r <= 2 and fm >= 4.5: return "Can't Lose Them"
        if r <= 2 and 2.5 < fm < 4.5: return "At Risk"
        if r <= 2 and fm <= 2.5: return "Hibernating"
        return "Others"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm.drop(columns=["FM_Avg"])


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    now = work["timestamp"].max()
    work["hour_sin"] = np.sin(2 * np.pi * work["timestamp"].dt.hour / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["timestamp"].dt.hour / 24)
    feat = work.groupby("customer_id_masked").agg(
        trx_count=("timestamp", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        recency_days=("timestamp", lambda x: (now - x.max()).days),
        active_days=("timestamp", lambda x: x.dt.date.nunique()),
        max_amount=("amount", "max"),
        min_amount=("amount", "min"),
        unique_merchants=("merchant_name", "nunique"),
        unique_mcc=("merchant_type", "nunique"),
        avg_hour_sin=("hour_sin", "mean"),
        avg_hour_cos=("hour_cos", "mean"),
        weekend_ratio=("timestamp", lambda x: (x.dt.dayofweek >= 5).mean()),
        night_ratio=("timestamp", lambda x: (x.dt.hour >= 18).mean()),
    ).reset_index()
    feat["cv_amount"] = feat["std_amount"] / feat["avg_amount"]
    feat["trx_per_day"] = feat["trx_count"] / feat["active_days"]
    feat["avg_spend_per_merchant"] = feat["total_amount"] / feat["unique_merchants"]
    return feat.replace([np.inf, -np.inf], np.nan)


def compact_records(df: pd.DataFrame, limit: int = 200) -> list[dict[str, Any]]:
    out = df.head(limit).copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M")
    return out.replace({np.nan: None}).to_dict("records")


def summary_payload(df: pd.DataFrame, source: str) -> dict[str, Any]:
    if df.empty:
        raise HTTPException(404, "No data found for the selected filters.")
    rfm = compute_rfm(df)
    rfm_agg = rfm.groupby("Segment").agg(
        Count=("Segment", "count"),
        Avg_Monetary=("Monetary", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Recency=("Recency", "mean"),
    ).reset_index()
    rfm_agg["Percentage"] = (rfm_agg["Count"] / rfm_agg["Count"].sum() * 100).round(1)

    daily = df.groupby(df["timestamp"].dt.date).agg(value=("amount", "sum"), transactions=("amount", "size")).reset_index()
    daily["time"] = daily["timestamp"].astype(str)
    hourly = df.groupby(df["timestamp"].dt.hour).agg(value=("amount", "sum"), transactions=("amount", "size")).reset_index()
    hourly = hourly.rename(columns={"timestamp": "hour"})

    return {
        "source": source,
        "filters": {
            "acquirers": sorted(df["acquirer"].dropna().astype(str).unique().tolist()),
            "issuers": sorted(df["issuer"].dropna().astype(str).unique().tolist()),
            "merchant_types": sorted(df["mcc_description"].dropna().astype(str).unique().tolist()),
            "min_date": str(df["timestamp"].min().date()),
            "max_date": str(df["timestamp"].max().date()),
        },
        "metrics": {
            "customers": int(df["customer_id_masked"].nunique()),
            "transactions": int(len(df)),
            "total_amount": safe_float(df["amount"].sum()),
            "avg_amount": safe_float(df["amount"].mean()),
        },
        "mcc_amount": df.groupby("mcc_description")["amount"].sum().sort_values(ascending=False).head(10).reset_index(name="amount").to_dict("records"),
        "top_merchants": df.groupby("merchant_name").agg(amount=("amount", "sum"), transactions=("amount", "size")).sort_values("amount", ascending=False).head(8).reset_index().to_dict("records"),
        "rfm": rfm_agg.sort_values("Count", ascending=False).replace({np.nan: None}).to_dict("records"),
        "rfm_detail": compact_records(rfm.sort_values("RFM-Score", ascending=False), 120),
        "daily": daily[["time", "value", "transactions"]].to_dict("records"),
        "hourly": hourly[["hour", "value", "transactions"]].to_dict("records"),
        "table": compact_records(df[["timestamp", "customer_id_masked", "merchant_name", "mcc_description", "acquirer", "issuer", "amount"]], 120),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "customer-profiling"}


@app.get("/categories")
def categories() -> dict[str, Any]:
    return {"categories": list(get_mcc_ranges().keys())}


@app.post("/insights")
def insights(payload: FilterPayload) -> dict[str, Any]:
    df, source = load_data(payload)
    return summary_payload(df, source)


@app.post("/cluster")
def cluster(payload: ClusterPayload) -> dict[str, Any]:
    df, source = load_data(payload)
    if df.empty:
        raise HTTPException(404, "No data found for clustering.")
    rfm = compute_rfm(df)
    selected = rfm.loc[rfm["Segment"].isin(payload.rfm_segments), "customer_id_masked"].unique()
    work = df[df["customer_id_masked"].isin(selected)].copy()
    if work["customer_id_masked"].nunique() < 5:
        raise HTTPException(422, "Aggregated data is too small for clustering.")

    cust = build_customer_features(work)
    numeric_features = [f for f in payload.features if f in cust.columns and pd.api.types.is_numeric_dtype(cust[f])]
    if len(numeric_features) < 2:
        raise HTTPException(422, "Select at least two numeric features.")

    x = cust[numeric_features].copy()
    imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=numeric_features)
    log_cols = [c for c in numeric_features if not any(token in c for token in ["sin", "cos", "ratio"])]
    imputed[log_cols] = np.log1p(imputed[log_cols])
    xs = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(xs)

    if payload.model_type == "KMeans":
        model = KMeans(n_clusters=payload.n_clusters, random_state=42, n_init=10)
    elif payload.model_type == "MiniBatch KMeans":
        model = MiniBatchKMeans(n_clusters=payload.n_clusters, batch_size=payload.batch_size, random_state=42, n_init=3)
    elif payload.model_type == "Gaussian Mixture":
        model = GaussianMixture(n_components=payload.n_clusters, random_state=42)
    elif payload.model_type == "DBSCAN":
        model = DBSCAN(eps=payload.eps, min_samples=payload.min_samples)
    else:
        model = HDBSCAN(min_cluster_size=payload.min_cluster_size)

    labels = model.fit_predict(xs)
    cust["cluster"] = [f"Cluster {int(label)}" if label != -1 else "Outlier" for label in labels]
    cust["x"] = z[:, 0]
    cust["y"] = z[:, 1]

    valid = labels != -1
    metrics: dict[str, float | int] = {
        "clusters_found": int(len(set(labels[valid]))),
        "outliers": int((~valid).sum()),
        "explained_variance_2d": round(float(pca.explained_variance_ratio_.sum()), 4),
    }
    if metrics["clusters_found"] > 1 and valid.sum() > 2:
        idx = np.random.RandomState(42).choice(np.where(valid)[0], min(10000, int(valid.sum())), replace=False)
        metrics["silhouette"] = round(float(silhouette_score(xs[idx], labels[idx])), 4)
        metrics["davies_bouldin"] = round(float(davies_bouldin_score(xs[valid], labels[valid])), 4)
        metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(xs[valid], labels[valid])), 4)

    share = cust.groupby("cluster")["total_amount"].sum().reset_index()
    share["percentage"] = (share["total_amount"] / share["total_amount"].sum() * 100).round(2)
    profile = cust.groupby("cluster")[numeric_features].mean().round(2).reset_index()

    spend = work.merge(cust[["customer_id_masked", "cluster"]], on="customer_id_masked", how="inner")
    spend["hour"] = spend["timestamp"].dt.hour
    spend["time_segment"] = pd.cut(
        spend["hour"],
        bins=[0, 6, 10, 12, 14, 17, 20, 23, 25],
        labels=["Night", "Early Morning", "Late Morning", "Early Afternoon", "Afternoon", "Evening", "Late Evening", "Night+"],
        right=False,
        include_lowest=True,
        ordered=False,
    ).astype(str).replace({"Night+": "Night"})
    spend["day_type"] = np.where(spend["timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday")
    spend["salary_type"] = np.where(spend["timestamp"].dt.day.isin(list(range(25, 32)) + list(range(1, 6))), "Salary Day", "Regular Day")

    def grouped(column: str) -> list[dict[str, Any]]:
        return spend.groupby([column, "cluster"]).agg(amount=("amount", "sum"), transactions=("amount", "size")).reset_index().to_dict("records")

    return {
        "source": source,
        "selected_customers": int(len(selected)),
        "features": numeric_features,
        "metrics": metrics,
        "pca": compact_records(cust[["customer_id_masked", "cluster", "x", "y", *numeric_features]], 2500),
        "share": share.to_dict("records"),
        "profile": profile.replace({np.nan: None}).to_dict("records"),
        "spending": {
            "time_segment": grouped("time_segment"),
            "day_type": grouped("day_type"),
            "salary_type": grouped("salary_type"),
        },
    }


@app.post("/chat")
def chat(payload: ChatPayload) -> dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "reply": "AI is not configured yet. Add GEMINI_API_KEY to backend/.env to enable the assistant. For now, use the RFM and clustering panels as your strategy baseline."
        }
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "You are an Artajasa marketing analyst. Answer concisely, practically, and based on the provided data. "
            f"Profile context: {payload.profile or {}}\nQuestion: {payload.message}"
        )
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as exc:
        return {"reply": f"Failed to load AI assistant: {exc}"}
