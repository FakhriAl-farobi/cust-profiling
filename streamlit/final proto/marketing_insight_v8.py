import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import math

from sklearn.mixture import GaussianMixture
from sklearn.cluster import (
    AgglomerativeClustering, KMeans, MiniBatchKMeans, DBSCAN,
    AffinityPropagation, MeanShift, SpectralClustering, OPTICS, Birch
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import google.generativeai as genai

try:
    import hdbscan
except ImportError:
    hdbscan = None

st.set_page_config(
    page_title="Artajasa Customer Profiling",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; }
h1, h2, h3 { color: #003153 !important; font-weight: 700; }
div[data-testid="stMetricValue"] { color: #005B96; font-size: 1.8rem; }
div[data-testid="stMetricLabel"] { color: #546E7A; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

def get_idx(candidates, cols):
    for c in candidates:
        if c in cols:
            return cols.index(c)
    return 0

def get_mcc_ranges():
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

genai.configure(api_key="")

def rename_cluster_agent(new_names: dict) -> str:
    if "cluster_mapping" not in st.session_state:
        st.session_state["cluster_mapping"] = {}
        
    for old_id, new_name in new_names.items():
        try:
            st.session_state["cluster_mapping"][int(old_id)] = new_name
        except ValueError:
            pass
            
    return "Penamaan cluster berhasil diperbarui."

agent_model = genai.GenerativeModel(
    'gemini-2.5-flash',
    tools=[rename_cluster_agent]
)

def build_customer_features(df, id_col):
    now = df["timestamp_local"].max()
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp_local'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp_local'].dt.hour / 24)

    feat = (
        df.groupby(id_col)
        .agg(
            trx_count=("timestamp_local", "count"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            std_amount=("amount", "std"),
            recency_days=("timestamp_local", lambda x: (now - x.max()).days),
            active_days=("timestamp_local", lambda x: x.dt.date.nunique()),
            max_amount=("amount", "max"),
            min_amount=("amount", "min"),
            unique_merchants=("merchant_name", "nunique"),
            unique_mcc=("merchant_type", "nunique"),
            avg_hour_sin=("hour_sin", "mean"),
            avg_hour_cos=("hour_cos", "mean"),
            weekend_ratio=("timestamp_local", lambda x: (x.dt.dayofweek >= 5).mean()),
            night_ratio=("timestamp_local", lambda x: (x.dt.hour >= 18).mean()),
        )
        .reset_index()
    )

    feat["cv_amount"] = feat["std_amount"] / feat["avg_amount"]
    feat["trx_per_day"] = feat["trx_count"] / feat["active_days"]
    feat["avg_spend_per_merchant"] = feat["total_amount"] / feat["unique_merchants"]

    return feat.replace([np.inf, -np.inf], np.nan)

@st.cache_resource
def get_db_connection():
    return st.connection("artajasa_db", type="sql")

conn = get_db_connection()

@st.cache_data(ttl=600)
def fetch_data_by_category(selected_labels, row_limit):
    if not selected_labels:
        return pd.DataFrame()

    conditions = []
    for label in selected_labels:
        start, end = get_mcc_ranges()[label]
        conditions.append(f"(CAST(merchant_type AS INTEGER) BETWEEN {start} AND {end})")

    query = f"""
        SELECT 
            acquirer, issuer, terminal_id, customer_id_masked, processing_code, 
            amount, merchant_pan, merchant_type, merchant_crit, merchant_name, 
            status_data, timestamp, mcc_description
        FROM data_transaksi_qris
        WHERE {" OR ".join(conditions)}
        LIMIT {row_limit}
    """
    
    df = conn.query(query)
    
    for c in df.columns:
        if "timestamp" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
            
    return df

def apply_memory_filters(df):
    st.sidebar.markdown("### Filter Lanjutan")
    df_f = df.copy()

    if "timestamp" in df_f.columns:
        min_d, max_d = df_f["timestamp"].min(), df_f["timestamp"].max()
        if pd.notna(min_d) and pd.notna(max_d):
            d1, d2 = st.sidebar.date_input("Periode Transaksi", [min_d.date(), max_d.date()])
            df_f = df_f[(df_f["timestamp"].dt.date >= d1) & (df_f["timestamp"].dt.date <= d2)]

    if "acquirer" in df_f.columns:
        opts = sorted(df_f["acquirer"].dropna().unique())
        sel = st.sidebar.multiselect("Acquirer", opts)
        if sel:
            df_f = df_f[df_f["acquirer"].isin(sel)]
    if "issuer" in df_f.columns:
        opts = sorted(df_f["issuer"].dropna().unique())
        sel = st.sidebar.multiselect("Issuer", opts)
        if sel:
            df_f = df_f[df_f["issuer"].isin(sel)]
    
    if "mcc_description" in df_f.columns:
        opts = sorted(df_f["mcc_description"].dropna().unique())
        sel = st.sidebar.multiselect("Merchant Type", opts)
        if sel:
            df_f = df_f[df_f["mcc_description"].isin(sel)]
    
    if "merchant_type" in df_f.columns:
        opts = sorted(df_f["merchant_type"].dropna().unique())
        sel = st.sidebar.multiselect("Merchant Category Codes", opts)
        if sel:
            df_f = df_f[df_f["merchant_type"].isin(sel)]
    return df_f

def compute_rfm(df, entity_col, date_col, amount_col):
    ref_date = df[date_col].max()
    
    rfm = (
        df.groupby(entity_col)
        .agg(
            Recency=(date_col, lambda x: (ref_date - x.max()).days),
            Frequency=(date_col, "count"),
            Monetary=(amount_col, "sum")
        )
        .reset_index()
    )

    def safe_qcut(s, q, asc=True):
        try:
            r = s.rank(method="first", ascending=asc)
            return pd.qcut(r, q, labels=False, duplicates="drop") + 1
        except:
            return pd.Series([1]*len(s))

    rfm["R-Score"] = safe_qcut(rfm["Recency"], 5, asc=False)
    rfm["F-Score"] = safe_qcut(rfm["Frequency"], 5)
    rfm["M-Score"] = safe_qcut(rfm["Monetary"], 5)
    
    rfm["RFM-Score"] = rfm["R-Score"] + rfm["F-Score"] + rfm["M-Score"]
    rfm["FM_Avg"] = ((rfm["F-Score"] + rfm["M-Score"]) / 2).round(1)

    def rfm_segment(row):
        r, fm = row["R-Score"], row["FM_Avg"]
        if r == 5 and fm >= 4: return "Champions"
        elif 3 <= r <= 4 and fm >= 4: return "Loyal Customers"
        elif r >= 4 and 2 <= fm <= 3.5: return "Potential Loyalist"
        elif r == 5 and fm < 2: return "New Customers"
        elif r == 4 and fm < 2: return "Promising"
        elif r == 3 and 2.5 < fm <= 3.5: return "Needs Attention"
        elif r == 3 and fm <= 2.5: return "About To Sleep"
        elif r <= 2 and fm >= 4.5: return "Can't Lose Them"
        elif r <= 2 and 2.5 < fm < 4.5: return "At Risk"
        elif r <= 2 and fm <= 2.5: return "Hibernating"
        else: return "Others"

    rfm["Segment"] = rfm.apply(rfm_segment, axis=1)
    return rfm.drop(columns=["FM_Avg"])

def temporal_aggregation(df, time_col, amount_col, by, metric):
    dff = df.copy()
    dff["time"] = dff[time_col].dt.hour if by == "hour" else dff[time_col].dt.date
    
    grp = dff.groupby("time")[amount_col]
    if metric == "freq": return dff.groupby("time").size().reset_index(name="value")
    if metric == "sum": return grp.sum().reset_index(name="value")
    if metric == "mean": return grp.mean().reset_index(name="value")
    if metric == "median": return grp.median().reset_index(name="value")
    return pd.DataFrame()

def analyze_data_structure(X_scaled, sample_size=2000):
    n_samples, n_features = X_scaled.shape
    sample_idx = np.random.choice(n_samples, min(n_samples, sample_size), replace=False)
    X_sample = X_scaled[sample_idx]

    pca_full = PCA().fit(X_scaled)
    explained_var_2d = np.sum(pca_full.explained_variance_ratio_[:2])
    is_spherical = explained_var_2d > 0.60

    mbk = MiniBatchKMeans(n_clusters=4, batch_size=256, random_state=42, n_init=3)
    mbk.fit(X_sample)
    
    distances = np.min(mbk.transform(X_sample), axis=1)
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    
    outlier_ratio = np.sum(distances > upper_bound) / len(distances)
    has_outliers = outlier_ratio > 0.05

    return is_spherical, has_outliers, explained_var_2d, outlier_ratio

def main():
    st.title("Customer Profiling Dashboard")

    selected_industries = st.sidebar.multiselect(
        "Sektor Industri",
        list(get_mcc_ranges().keys()),
        default=list(get_mcc_ranges().keys())[:2]
    )
    
    cpan_limit = st.sidebar.select_slider(
        "Batas Sampel (Jumlah CPAN)",
        [1000, 3000, 5000, 10000, 20000, 50000, 100000],
        value=5000
    )

    if not selected_industries:
        st.warning("Silakan pilih minimal satu Sektor Industri.")
        st.stop()

    df_raw = fetch_data_by_category(selected_industries, row_limit=5_000_000)
    
    if df_raw.empty:
        st.error("Data tidak ditemukan untuk kategori ini.")
        st.stop()

    unique_cpan = df_raw["customer_id_masked"].dropna().unique()

    sample_size = min(cpan_limit, len(unique_cpan))

    sampled_cpan = np.random.RandomState(42).choice(
        unique_cpan,
        size=sample_size,
        replace=False
    )

    df_raw = df_raw[df_raw["customer_id_masked"].isin(sampled_cpan)]
    
    df_view = apply_memory_filters(df_raw)
    
    st.metric("Jumlah CPAN", df_view["customer_id_masked"].nunique())
    st.metric("Jumlah Transaksi", len(df_view))

    amt_col = "amount"
    date_col = "timestamp"
    entity_col = "customer_id_masked"

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "RFM", "Advanced Clustering", "Spending Pattern", "AI Chatbot"])
    with tab1:
        st.dataframe(df_view, use_container_width=True, height=500)
        st.divider()
        chart = df_view.groupby("merchant_type")[amt_col].sum().reset_index()
        st.bar_chart(chart, x="merchant_type", y=amt_col)

    with tab2:
        st.subheader("RFM Segmentation")
        
        rfm = compute_rfm(df_view, entity_col, date_col, amt_col)
        rfm_global = rfm.copy() 

        rfm_agg = (
            rfm.groupby("Segment")
            .agg(
                Count=("Segment", "count"),
                Avg_Monetary=("Monetary", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Recency=("Recency", "mean")
            )
            .reset_index()
        )

        total_customers = rfm_agg["Count"].sum()
        rfm_agg["Percentage"] = (rfm_agg["Count"] / total_customers * 100).round(1)

        segment_colors = {
            "Champions": "#00C853", "Loyal Customers": "#2962FF", "Potential Loyalist": "#00B0FF",
            "New Customers": "#00E5FF", "Promising": "#FFD600", "Needs Attention": "#FFAB00",
            "About To Sleep": "#FF6D00", "At Risk": "#D50000", "Can't Lose Them": "#AA00FF",
            "Hibernating": "#455A64", "Others": "#90A4AE"
        }

        c_chart, c_table = st.columns([1.8, 1.2])

        with c_chart:
            st.caption("Visualisasi Proporsi Setiap Segmen")
            
            fig_treemap = px.treemap(
                rfm_agg, path=["Segment"], values="Count", color="Segment",
                color_discrete_map=segment_colors, custom_data=["Percentage", "Avg_Monetary"],
                title=f"Total Customers: {total_customers:,.0f}"
            )

            fig_treemap.update_traces(
                textinfo="label+value+percent entry",
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Porsi: %{customdata[0]}%<br>Rata-rata Tx: %{customdata[1]:,.0f}"
            )
            
            fig_treemap.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=450)
            st.plotly_chart(fig_treemap, use_container_width=True)

        with c_table:
            st.caption("Detail Metrik per Segmen")
            st.dataframe(
                rfm_agg.sort_values("Count", ascending=False),
                column_order=("Segment", "Count", "Avg_Frequency", "Avg_Monetary", "Avg_Recency"),
                column_config={
                    "Segment": "Customer Segment",
                    "Count": st.column_config.NumberColumn("Customers", format="%d 👤"),
                    "Avg_Frequency": st.column_config.ProgressColumn("Avg Freq", format="%.1f x", min_value=0, max_value=float(rfm_agg["Avg_Frequency"].max()) * 1.1),
                    "Avg_Monetary": st.column_config.NumberColumn("Avg Value (Rp)", format="Rp %d"),
                    "Avg_Recency": st.column_config.NumberColumn("Avg Recency", format="%d hari")
                },
                hide_index=True, use_container_width=True, height=450
            )

        with st.expander("Lihat Data Detail per User (Raw RFM Score)"):
            st.dataframe(rfm.sort_values("RFM-Score", ascending=False), use_container_width=True)

        st.divider()
        st.subheader("Rekomendasi Strategi Marketing per Segmen")

        recommendations = {
            "Champions": """
            **Action:** Introduce new and upcoming products and drops. Reward them and help them share updates.
            **Message:** "Hi {Name}, as our VVIP, you get early access to our latest drops! Grab yours before the public launch."
            """,
            "Loyal Customers": """
            **Action:** Upsell higher value products. Ask for reviews. Engage them.
            **Message:** "Loving your recent buys? Try our Premium Add-on for even better results. Use code LOYAL20 for 20% off!"
            """,
            "Potential Loyalist": """
            **Action:** Offer membership / loyalty program, recommend other products.
            **Message:** "You're on a roll! Join our Member Club today and earn points on every purchase. Click here to unlock benefits."
            """,
            "New Customers": """
            **Action:** Provide on-boarding support, give them early success, start building relationship.
            **Message:** "Welcome aboard! Here’s a quick guide to get the most out of your first purchase. Need help? We're here 24/7."
            """,
            "Promising": """
            **Action:** Check on their need for replenishment, ask for feedback and share the most popular products.
            **Message:** "Running low? It’s time to restock your favorites. Order now to ensure you don't run out!"
            """,
            "Needs Attention": """
            **Action:** Make limited time offers, Recommend based on past purchases. Reactivate them.
            **Message:** "We miss you! Here is a special voucher valid only for 48 hours. Come back and treat yourself."
            """,
            "About To Sleep": """
            **Action:** Share valuable resources, recommend popular products at discount, reconnect with them.
            **Message:** "It's been a while. See what's trending this week—we think you'll love these new arrivals (now 15% off)."
            """,
            "At Risk": """
            **Action:** Send personalized emails to reconnect, offer renewals, provide helpful resources.
            **Message:** "Is everything okay? We noticed you haven't visited lately. Let us know how we can improve your experience."
            """,
            "Can't Lose Them": """
            **Action:** Remind them about the reasons they loved your brand in the first place, rekindle the relationship.
            **Message:** "We want you back! Here is our biggest offer of the year: 40% OFF your next purchase. Valid for you only."
            """,
            "Hibernating": """
            **Action:** Offer other relevant products and special discounts. Recreate brand value.
            **Message:** "Long time no see! We have a special welcome back gift waiting for you. Check it out here."
            """,
            "Others": "Keep engaging with general brand updates and newsletters."
        }

        active_segments = sorted(rfm_agg["Segment"].unique())
        selected_seg = st.selectbox("Pilih Segmen:", active_segments)

        if selected_seg in recommendations:
            st.info(recommendations[selected_seg])

    with tab3:
        st.subheader("Customizing Customer Profiling")

        seg_opts = sorted(rfm_global["Segment"].unique())
        selected_segs = st.multiselect(
            "Pilih RFM Segment",
            seg_opts,
            default=[x for x in ["Hibernating", "At Risk", "About To Sleep"] if x in seg_opts]
        )

        if not selected_segs:
            st.warning("Pilih minimal 1 RFM segment")
            st.stop()

        filtered_cpan = rfm_global.loc[rfm_global["Segment"].isin(selected_segs), entity_col].unique()
        st.info(f"Menggunakan {len(filtered_cpan)} CPAN dari segmen terpilih.")

        if len(filtered_cpan) > 50_000:
            st.error("Terlalu banyak data. Persempit segmen.")
            st.stop()

        work = df_view[df_view[entity_col].isin(filtered_cpan)].copy()
        
        cols = work.columns.tolist()
        c1, c2, c3 = st.columns(3)
        id_col_sel = c1.selectbox("ID Column", cols, index=get_idx(["customer_id_masked"], cols))
        ts_col_sel = c2.selectbox("Timestamp Column", cols, index=get_idx(["timestamp"], cols))
        amt_col_sel = c3.selectbox("Amount Column", cols, index=get_idx(["amount"], cols))

        work = work.dropna(subset=[id_col_sel, ts_col_sel, amt_col_sel])
        work["timestamp_local"] = pd.to_datetime(work[ts_col_sel], errors="coerce")
        work["amount"] = pd.to_numeric(work[amt_col_sel], errors="coerce")

        cust = build_customer_features(work, id_col_sel)
        
        if len(cust) < 5:
            st.warning("Data aggregasi terlalu sedikit.")
            st.stop()

        feats = st.multiselect(
            "Features used for Clustering",
            [c for c in cust.columns if c != id_col_sel],
            default=["trx_count", "total_amount", "recency_days", "avg_hour_sin", "avg_hour_cos"]
        )

        if not feats:
            st.error("Pilih fitur terlebih dahulu.")
            st.stop()

        if len(feats) < 2:
            st.warning("Minimal 2 fitur diperlukan untuk clustering yang stabil.")
            st.stop()
        
        X = cust[feats].copy()
        X = X.select_dtypes(include=[np.number])
        X = X.dropna(axis=1, how="all")

        current_feats = X.columns.tolist()

        imputer = SimpleImputer(strategy="median")
        X_imp_arr = imputer.fit_transform(X)

        if X_imp_arr.shape[1] != len(current_feats):
            current_feats = imputer.get_feature_names_out(current_feats).tolist()

        feats = current_feats
        X_imp = pd.DataFrame(X_imp_arr, columns=feats)

        cols_to_log = [c for c in feats if not any(x in c for x in ['sin', 'cos', 'ratio'])]
        if cols_to_log:
            X_imp[cols_to_log] = np.log1p(X_imp[cols_to_log])

        Xs = StandardScaler().fit_transform(X_imp)

        pca_vis = PCA(n_components=2, random_state=42)
        Z = pca_vis.fit_transform(Xs)
        cust["x"], cust["y"] = Z[:, 0], Z[:, 1]
        
        st.markdown("### Distribusi Data (PCA)")
        
        fig_pre = px.scatter(
            cust, x="x", y="y", 
            hover_data=feats,
            title="Proyeksi Data (PCA) - Sebelum Clustering",
            opacity=0.6
        )
        st.plotly_chart(fig_pre, use_container_width=True)
        st.divider()

        st.markdown("### Konfigurasi Model")
        f1, f2 = st.columns([1.5, 2.5])
        
        with f1:
            is_spherical, has_outliers, exp_var, out_ratio = analyze_data_structure(Xs)

            st.markdown("#### Smart System Analysis")
            
            col_a, col_b = st.columns(2)
            col_a.metric("PCA Explained Variance (2D)", f"{exp_var:.1%}")
            col_b.metric("Indikasi Outlier", f"{out_ratio:.1%}")

            available_models = []

            if has_outliers or not is_spherical:
                st.warning("Terdeteksi struktur data kompleks atau anomali ekstrem. Density-based direkomendasikan.")
                if len(Xs) <= 20000:
                    available_models.extend(["HDBSCAN", "DBSCAN"])
                else:
                    available_models.append("HDBSCAN")

                if len(Xs) <= 50000: 
                    available_models.append("Gaussian Mixture")
                    
                if not has_outliers and len(Xs) <= 5000:
                    available_models.append("MeanShift")
                
                available_models.append("MiniBatch KMeans") 
            else:
                st.success("Struktur stabil (Spherical). Centroid-based aman digunakan.")
                available_models.extend(["KMeans", "MiniBatch KMeans", "Gaussian Mixture"])
                if len(Xs) <= 5000:
                    available_models.append("Agglomerative (Hierarchical)")

            model_type = st.selectbox("Algoritma Terpilih", available_models)

            k, batch_size, eps = 4, 1024, 0.5
            cov_type, link_type, damping, bandwidth, min_cluster_size = "full", "ward", 0.9, 2.0, 20
            
            dynamic_min_samples = max(5, len(feats) + 1)
            min_samples = dynamic_min_samples

            if model_type in ["KMeans", "MiniBatch KMeans", "Agglomerative (Hierarchical)", "Gaussian Mixture"]:
                k = st.number_input("n_clusters (K)", 2, 20, 4)

            if model_type == "MiniBatch KMeans":
                batch_size = st.number_input("batch_size", 256, 10000, 1024, step=256)

            if model_type == "Gaussian Mixture":
                cov_type = st.selectbox("covariance_type", ["full", "diag", "tied", "spherical"])

            if model_type == "Agglomerative (Hierarchical)":
                link_type = st.selectbox("linkage", ["ward", "average", "complete"])

            if model_type == "DBSCAN":
                eps = st.slider("eps (Epsilon Distance)", 0.1, 5.0, 0.5)
                min_samples = st.slider("min_samples", 2, 50, dynamic_min_samples)

            if model_type == "HDBSCAN":
                min_cluster_size = st.slider("min_cluster_size", 5, 200, dynamic_min_samples * 2)
                
            if model_type == "MeanShift":
                bandwidth = st.slider("bandwidth", 0.1, 10.0, 2.0)

            available_metrics = {
                "Silhouette Score": "silhouette",
                "Davies-Bouldin Index": "davies_bouldin",
                "Calinski-Harabasz Index": "calinski_harabasz"
            }
            selected_metric_names = st.multiselect(
                "Evaluation Metrics",
                options=list(available_metrics.keys()),
                default=["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"][:2]
            )

            show_elbow = model_type in ["KMeans", "MiniBatch KMeans"]
            run = st.button("Jalankan Profiling", type="primary", use_container_width=True)

        with f2:
            if show_elbow:
                st.markdown("##### Analisis Optimal K (Elbow Method)")
                k_min_elb, k_max_elb = st.slider("Range K untuk Test", 2, 12, (2, 8))
                
                if st.button("Hitung Rekomendasi K"):
                    ks_list, inertias, sils = [], [], []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_iter = k_max_elb - k_min_elb + 1
                    for i, tmp_k in enumerate(range(k_min_elb, k_max_elb + 1)):
                        progress = (i + 1) / total_iter
                        progress_bar.progress(progress)
                        status_text.text(f"Computing K={tmp_k}...")

                        km_test = KMeans(n_clusters=tmp_k, random_state=42, n_init=5)
                        lbls_test = km_test.fit_predict(Xs)
                        
                        ks_list.append(tmp_k)
                        inertias.append(km_test.inertia_)
                        
                        if len(set(lbls_test)) > 1:
                            if len(Xs) > 10000:
                                idx_samp = np.random.choice(len(Xs), 10000, replace=False)
                                score = silhouette_score(Xs[idx_samp], lbls_test[idx_samp])
                            else:
                                score = silhouette_score(Xs, lbls_test)
                            sils.append(score)
                        else:
                            sils.append(-1)
                    
                    progress_bar.empty()
                    status_text.empty()

                    best_idx = np.argmax(sils)
                    best_k_math = ks_list[best_idx]
                    best_score = sils[best_idx]

                    st.success(f"**Rekomendasi Jumlah Cluster:** K = {best_k_math} (Silhouette Score: {best_score:.3f})")
                    
                    ec1, ec2 = st.columns(2)
                    fig_elb = px.line(x=ks_list, y=inertias, title="Elbow Method", labels={"x":"K", "y":"Inertia"})
                    ec1.plotly_chart(fig_elb, use_container_width=True)

                    fig_sil = px.line(x=ks_list, y=sils, title="Silhouette Score", labels={"x":"K", "y":"Score"})
                    ec2.plotly_chart(fig_sil, use_container_width=True)

        if run:
            with st.spinner(f"Running {model_type}..."):
                labels = None
                
                if model_type == "KMeans": model = KMeans(n_clusters=k, random_state=42, n_init=10)
                elif model_type == "MiniBatch KMeans": model = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)
                elif model_type == "Gaussian Mixture": model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)
                elif model_type == "Affinity Propagation": model = AffinityPropagation(damping=damping, random_state=42)
                elif model_type == "MeanShift": model = MeanShift(bandwidth=bandwidth)
                elif model_type == "Spectral Clustering": model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
                elif model_type == "Agglomerative (Hierarchical)": model = AgglomerativeClustering(n_clusters=k, linkage=link_type)
                elif model_type == "DBSCAN": model = DBSCAN(eps=eps, min_samples=min_samples)
                elif model_type == "OPTICS": model = OPTICS(min_samples=min_samples, max_eps=eps)
                elif model_type == "HDBSCAN": model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                elif model_type == "BIRCH": model = Birch(n_clusters=k)

                try:
                    labels = model.fit_predict(Xs)
                except Exception as e:
                    st.error(f"Error during fitting: {e}")
                    st.stop()

                st.session_state["saved_labels"] = labels
                st.session_state["saved_model_type"] = model_type
                st.session_state["saved_feats"] = feats

        if "saved_labels" in st.session_state and st.session_state.get("saved_feats") == feats:
            labels = st.session_state["saved_labels"]
            used_model_type = st.session_state["saved_model_type"]

            st.divider()
            st.markdown("### Definisikan Nama Cluster")
            
            unique_labels_all = sorted(list(set(labels)))
            
            if "cluster_mapping" not in st.session_state:
                st.session_state["cluster_mapping"] = {cl_id: str(cl_id) for cl_id in unique_labels_all}
                
            for cl_id in unique_labels_all:
                if cl_id not in st.session_state["cluster_mapping"]:
                    st.session_state["cluster_mapping"][cl_id] = str(cl_id)

            num_cols = max(1, min(len(unique_labels_all), 5))
            c_inputs = st.columns(num_cols)
            for i, cl_id in enumerate(unique_labels_all):
                with c_inputs[i % num_cols]:
                    new_val = st.text_input(f"Nama Cluster {cl_id}", value=st.session_state["cluster_mapping"][cl_id], key=f"map_{cl_id}")
                    st.session_state["cluster_mapping"][cl_id] = new_val

            mapped_labels = [st.session_state["cluster_mapping"][l] for l in labels]
            cust["cluster"] = mapped_labels
            st.session_state["clustered_data"] = cust[[id_col_sel, "cluster"]].copy()

            mask_valid = labels != -1
            Xs_valid = Xs[mask_valid]
            labels_valid = labels[mask_valid]
            
            n_clusters_found = len(set(labels_valid))
            n_outliers = np.sum(~mask_valid)
            
            calculated_metrics = {}
            if n_clusters_found > 1 and len(Xs_valid) > 0:
                idx_metric = np.random.choice(len(Xs_valid), min(10000, len(Xs_valid)), replace=False)
                for name in selected_metric_names:
                    key = available_metrics[name]
                    if key == "silhouette": calculated_metrics[name] = silhouette_score(Xs_valid[idx_metric], labels_valid[idx_metric])
                    elif key == "davies_bouldin": calculated_metrics[name] = davies_bouldin_score(Xs_valid, labels_valid)
                    elif key == "calinski_harabasz": calculated_metrics[name] = calinski_harabasz_score(Xs_valid, labels_valid)
            
            st.divider()
            st.markdown("### Hasil Clustering")
            
            cols = st.columns(2 + len(calculated_metrics))
            cols[0].metric("Clusters Found", n_clusters_found)
            cols[1].metric("Outliers Detected", n_outliers)
            
            for idx, (metric_name, value) in enumerate(calculated_metrics.items()):
                cols[idx+2].metric(metric_name, f"{value:.3f}")

            if n_clusters_found <= 1:
                st.info("Hanya ditemukan satu cluster valid (atau semua noise). Metrik dievaluasi dilewati.")

            c_share, c_pca = st.columns([2, 1])

            with c_pca:
                st.plotly_chart(
                    px.scatter(
                        cust, x="x", y="y", 
                        color=cust["cluster"].astype(str),
                        hover_data=feats,
                        title=f"Proyeksi Data (PCA) - Setelah {used_model_type}",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    ),
                    use_container_width=True
                )

            with c_share:
                st.subheader("Market Share")
                amt_share = cust.groupby("cluster")["total_amount"].sum().reset_index()
                amt_share["percentage (%)"] = (amt_share["total_amount"] / amt_share["total_amount"].sum()) * 100
                
                st.plotly_chart(
                    px.bar(amt_share, x="cluster", y="percentage (%)", text_auto=".1f", 
                        title="Value (% Rp)", height=350),
                    use_container_width=True
                )

            st.divider()
            st.subheader("Customer Profile Results")
            profile = cust.groupby("cluster")[feats].mean().reset_index()
            def get_hour_from_sincos(row):
                sin_val = row.get("avg_hour_sin")
                cos_val = row.get("avg_hour_cos")
                
                if pd.isna(sin_val) or pd.isna(cos_val):
                    return np.nan

                angle = np.arctan2(sin_val, cos_val)
                hour_float = (angle / (2 * np.pi)) * 24

                if hour_float < 0:
                    hour_float += 24
                    
                hour = int(hour_float)
                minute = int((hour_float - hour) * 60)

                return f"{hour:02d}:{minute:02d}"

            if 'avg_hour_sin' in profile.columns and 'avg_hour_cos' in profile.columns:
                profile['avg_hour'] = profile.apply(get_hour_from_sincos, axis=1)
                
                cols_p = list(profile.columns)
                target_idx = cols_p.index('avg_hour_cos') + 1
                cols_p.insert(target_idx, cols_p.pop(cols_p.index('avg_hour')))

                profile = profile[cols_p]
                profile.drop(columns=['avg_hour_sin', 'avg_hour_cos'], inplace=True)

            st.dataframe(
                profile.style.background_gradient(cmap="GnBu").format(precision=2), 
                use_container_width=True
            )   

    with tab4:
        st.subheader("Cluster Spending Pattern Analysis")
        if "clustered_data" not in st.session_state:
            st.warning("Silakan jalankan 'Run Clustering' di tab Advanced Clustering terlebih dahulu.")
        else:
            df_spend = df_view.merge(
                st.session_state["clustered_data"], 
                left_on=entity_col, 
                right_on=id_col_sel, 
                how="inner"
            )
            
            bins = [0, 6, 10, 12, 14, 17, 20, 23, 25]
            labels = [
                "Night Time (23:00-05:59)", 
                "Early Morning (06:00-09:59)", 
                "Late Morning (10:00-11:59)", 
                "Early Afternoon (12:00-13:59)", 
                "Afternoon (14:00-16:59)", 
                "Evening (17:00-19:59)", 
                "Late Evening (20:00-22:59)",
                "Night Time (23:00-05:59) "
            ]

            df_spend['hour_actual'] = df_spend[date_col].dt.hour
            df_spend['time_segment'] = pd.cut(
                df_spend['hour_actual'], 
                bins=bins, 
                labels=labels, 
                right=False, 
                include_lowest=True,
                ordered=False
            )

            df_spend['is_weekend'] = np.where(df_spend[date_col].dt.dayofweek >= 5, "Weekend", "Weekday")
            df_spend['is_salary_day'] = np.where(
                df_spend[date_col].dt.day.isin(list(range(25, 32)) + list(range(1, 6))), 
                "Salary Day", 
                "Regular Day"
            )
            df_spend['is_night'] = np.where(df_spend[date_col].dt.hour >= 18, "Night Spender", "Day Spender")
            
            avail_clusters = sorted(df_spend["cluster"].unique())
            sel_clusters = st.multiselect("Pilih Cluster", avail_clusters, default=avail_clusters)
            
            metric_spend = st.radio("Metrik", ["Frequency", "Average Amount", "Sum Amount"], horizontal=True)            
            df_sp_filt = df_spend[df_spend["cluster"].isin(sel_clusters)].copy()

            if metric_spend == "Frequency":
                hf = "count"
                ya = None
            elif metric_spend == "Sum Amount":
                hf = "sum"
                ya = "amount"
            else: 
                hf = "avg"
                ya = "amount"

            st.divider()
            sc1, sc2 = st.columns(2)
            
            with sc1:
                fig_time = px.histogram(
                    df_sp_filt, x="time_segment", y=ya, color="cluster", 
                    histfunc=hf, barmode="group", title="Spending Pattern by Time Segment",
                    category_orders={"time_segment": labels[:-1]}
                )
                st.plotly_chart(fig_time, use_container_width=True)
                
                fig_weekend = px.histogram(df_sp_filt, x="is_weekend", y=ya, color="cluster", histfunc=hf, barmode="group", title="Weekend vs Weekday")
                st.plotly_chart(fig_weekend, use_container_width=True)
                
            with sc2:
                fig_salary = px.histogram(df_sp_filt, x="is_salary_day", y=ya, color="cluster", histfunc=hf, barmode="group", title="Salary Day vs Regular Day")
                st.plotly_chart(fig_salary, use_container_width=True)
                
                fig_night = px.histogram(df_sp_filt, x="is_night", y=ya, color="cluster", histfunc=hf, barmode="group", title="Night vs Day Spender")
                st.plotly_chart(fig_night, use_container_width=True)
            
            st.divider()
            st.subheader("Deep Dive Time Analysis")
            
            df_sp_filt['month'] = df_sp_filt[date_col].dt.month_name()
            months_order = ["January", "February", "March", "April", "May", "June", 
                            "July", "August", "September", "October", "November", "December"]
            
            avail_months = [m for m in months_order if m in df_sp_filt['month'].unique()]
            
            fig_month = px.histogram(
                df_sp_filt, x="month", y=ya, color="cluster", 
                histfunc=hf, barmode="group", title="Monthly Distribution Analysis",
                category_orders={"month": months_order}
            )
            st.plotly_chart(fig_month, use_container_width=True)

            do_date_dive = st.checkbox("Deep Dive per Tanggal di Bulan Tertentu?")
            if do_date_dive:
                sel_month = st.selectbox("Pilih Bulan untuk Bedah Tanggal", avail_months)
                df_month = df_sp_filt[df_sp_filt['month'] == sel_month].copy()
                df_month['day'] = df_month[date_col].dt.day
                
                pd_agg_func = "mean" if hf == "avg" else hf 
                
                df_daily = df_month.groupby(['day', 'cluster'])[amt_col if ya else id_col_sel].agg(pd_agg_func).reset_index(name='val')
                
                fig_day = px.line(
                    df_daily, x="day", y="val", color="cluster",
                    title=f"Daily Transaction Trend - {sel_month}", markers=True
                )
                st.plotly_chart(fig_day, use_container_width=True)
            
            st.markdown("### Peak Summary")
            res_date = df_sp_filt.groupby(df_sp_filt[date_col].dt.date).size()
            if not res_date.empty:
                peak_val = res_date.max()
                peak_day = res_date.idxmax()
                
                c_p1, c_p2 = st.columns(2)
                c_p1.metric("Peak Transactions (All Time)", f"{peak_val:,} Trx")
                c_p2.metric("Peak Date Observed", str(peak_day))

    with tab5:
        st.subheader("AI Assistant")
        
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = agent_model.start_chat(enable_automatic_function_calling=True)
            st.session_state.chat_history_ui = []

        for msg in st.session_state.chat_history_ui:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(""):
            st.session_state.chat_history_ui.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            context_string = "Belum ada data cluster yang digenerate."
            if "clustered_data" in st.session_state and 'profile' in locals():
                context_string = f"Current Cluster Mapping: {st.session_state.get('cluster_mapping', {})}. Profile data: {profile.to_dict()}"

            mapping_before = st.session_state.get("cluster_mapping", {}).copy()

            with st.chat_message("assistant"):
                with st.spinner("Menyiapkan respons..."):
                    try:
                        full_prompt = f"Konteks Data Profiling: {context_string}\n\nInstruksi: Jika user menghendaki ubah nama cluster, panggil fungsi rename_cluster_agent. Pertanyaan User: {prompt}"
                        response = st.session_state.chat_session.send_message(full_prompt)
                        st.markdown(response.text)
                        st.session_state.chat_history_ui.append({"role": "assistant", "content": response.text})
                        
                        mapping_after = st.session_state.get("cluster_mapping", {})
                        if mapping_before != mapping_after:
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Gagal memuat AI: {e}")

if __name__ == "__main__":
    main()
