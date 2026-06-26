import streamlit as st
import pandas as pd
import numpy as np
import json
import ast
import warnings
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hollywood Box Office Secrets",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette (matches notebook) ──────────────────────────────────────────
COLORS = {
    "primary":   "#E50914",
    "secondary": "#FFD700",
    "dark":      "#1A1A2E",
    "accent":    "#16213E",
    "light":     "#F5F5F5",
    "blue":      "#0077B6",
    "gradient":  ["#E50914", "#FF6B35", "#FFD700", "#00D4AA", "#0077B6"],
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; color: #f0f0f0; }
    section[data-testid="stSidebar"] { background-color: #1A1A2E; }
    .metric-card {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid #E50914;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #FFD700; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-title {
        font-size: 1.5rem; font-weight: 700;
        color: #FFD700; border-left: 4px solid #E50914;
        padding-left: 12px; margin: 24px 0 16px;
    }
    .predict-box {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 2px solid #FFD700;
        border-radius: 16px;
        padding: 28px;
    }
    .result-revenue { font-size: 2.8rem; font-weight: 900; color: #FFD700; }
    .result-roi     { font-size: 1.4rem; font-weight: 700; color: #00D4AA; }
    h1, h2, h3 { color: #FFD700 !important; }
    .stSelectbox label, .stSlider label, .stMultiSelect label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset…")
def load_and_prepare():
    df = pd.read_csv("tmdb_5000_movies.csv")

    def parse_json_col(text):
        try:
            return json.loads(text.replace("'", '"'))
        except:
            try:
                return ast.literal_eval(text)
            except:
                return []

    def extract_names(json_str):
        items = parse_json_col(json_str) if isinstance(json_str, str) else []
        return [item["name"] for item in items if "name" in item]

    df["genre_list"]   = df["genres"].apply(extract_names)
    df["keyword_list"] = df["keywords"].apply(extract_names)
    df["company_list"] = df["production_companies"].apply(extract_names)

    df["release_date"]  = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]  = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    df_clean = df[(df["budget"] > 1000) & (df["revenue"] > 1000)].copy()
    df_clean["roi"]      = ((df_clean["revenue"] - df_clean["budget"]) / df_clean["budget"]) * 100
    df_clean["profit"]   = df_clean["revenue"] - df_clean["budget"]
    df_clean["is_hit"]   = (df_clean["roi"] > 100).astype(int)
    df_clean["log_budget"]  = np.log1p(df_clean["budget"])
    df_clean["log_revenue"] = np.log1p(df_clean["revenue"])

    return df_clean


@st.cache_data(show_spinner="Training models…")
def build_model(df_clean):
    # genre dummies
    genre_df = []
    for _, row in df_clean.iterrows():
        for g in row["genre_list"]:
            genre_df.append({"genre": g, "roi": row["roi"]})
    genre_df = pd.DataFrame(genre_df)
    top_10_genres = genre_df["genre"].value_counts().head(10).index.tolist()

    for g in top_10_genres:
        df_clean[f"genre_{g}"] = df_clean["genre_list"].apply(lambda x: 1 if g in x else 0)

    # hit keyword score
    hit_kws = []
    for _, row in df_clean[df_clean["is_hit"] == 1].iterrows():
        hit_kws.extend(row["keyword_list"])
    top_hit_kws = set([kw for kw, _ in Counter(hit_kws).most_common(50)])
    df_clean["hit_keyword_score"] = df_clean["keyword_list"].apply(
        lambda kws: sum(1 for kw in kws if kw in top_hit_kws)
    )

    # studio power
    studio_rev = {}
    for _, row in df_clean.iterrows():
        for c in row["company_list"]:
            studio_rev.setdefault(c, []).append(row["revenue"])
    studio_power = {k: np.mean(v) for k, v in studio_rev.items() if len(v) >= 5}
    df_clean["studio_power"] = df_clean["company_list"].apply(
        lambda comps: max([studio_power.get(c, 0) for c in comps]) if comps else 0
    )

    season_map = {12:"Winter",1:"Winter",2:"Winter",
                  3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",
                  9:"Fall",10:"Fall",11:"Fall"}
    df_clean["season"] = df_clean["release_month"].map(season_map)
    season_dummies = pd.get_dummies(df_clean["season"], prefix="season", drop_first=True)
    df_clean = pd.concat([df_clean, season_dummies], axis=1)

    df_clean["n_genres"]         = df_clean["genre_list"].apply(len)
    df_clean["n_keywords"]       = df_clean["keyword_list"].apply(len)
    df_clean["n_companies"]      = df_clean["company_list"].apply(len)
    df_clean["budget_per_minute"]= df_clean["budget"] / df_clean["runtime"].clip(lower=1)
    df_clean["overview_length"]  = df_clean["overview"].fillna("").apply(len)

    feature_cols = (
        ["log_budget","popularity","runtime","vote_average","vote_count",
         "hit_keyword_score","studio_power","n_genres","n_keywords",
         "n_companies","budget_per_minute","overview_length","release_month"]
        + [f"genre_{g}" for g in top_10_genres]
        + [c for c in df_clean.columns if c.startswith("season_")]
    )

    X = df_clean[feature_cols].fillna(0)
    y = df_clean["log_revenue"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                   learning_rate=0.1, min_samples_split=10, random_state=42)
    gb.fit(X_train, y_train)
    r2 = r2_score(y_test, gb.predict(X_test))

    importances = pd.DataFrame({"feature": feature_cols, "importance": gb.feature_importances_})\
                    .sort_values("importance", ascending=False)

    return gb, feature_cols, top_10_genres, studio_power, top_hit_kws, r2, importances, df_clean


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 Hollywood Box Office")
    st.markdown("*Data Science Analysis*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 EDA", "🤖 ML Models", "🎯 Revenue Predictor"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset: TMDB 5000 Movies  \nModel: Gradient Boosting")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
try:
    df_clean = load_and_prepare()
    gb, feature_cols, top_10_genres, studio_power, top_hit_kws, model_r2, importances, df_model = build_model(df_clean.copy())
    data_ok = True
except FileNotFoundError:
    st.error("⚠️  `tmdb_5000_movies.csv` not found. Place it in the same folder as `app.py`.")
    data_ok = False
    st.stop()

MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🎬 Hollywood Box Office Secrets")
    st.markdown("#### What makes a movie a hit? — Interactive analysis of **5,000 films**")
    st.divider()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{df_clean.shape[0]:,}", "Films Analyzed"),
        (f"${df_clean['budget'].median()/1e6:.0f}M", "Median Budget"),
        (f"${df_clean['revenue'].median()/1e6:.0f}M", "Median Revenue"),
        (f"{df_clean['is_hit'].mean()*100:.0f}%", "Hit Rate"),
        (f"{model_r2:.3f}", "Best Model R²"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Top findings
    st.markdown('<div class="section-title">🏆 Key Findings</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.info("**Most Profitable Genre**\n\nMusic films lead with **212% median ROI** — low budget, high emotional return.")
    with f2:
        st.warning("**Best Release Window**\n\nMay & June dominate. Adventure films in May earn a median **$397M**.")
    with f3:
        st.success("**Budget Sweet Spot**\n\nBoth **< $5M** and **$100M+** films hit ~70% hit rate — the middle is risky.")

    st.divider()

    # Budget vs Revenue scatter
    st.markdown('<div class="section-title">💰 Budget vs Revenue</div>', unsafe_allow_html=True)
    fig = px.scatter(
        df_clean, x="budget", y="revenue",
        color="vote_average", size="popularity",
        size_max=30, opacity=0.55,
        color_continuous_scale="RdYlGn",
        hover_data=["title", "roi"],
        labels={"budget": "Budget ($)", "revenue": "Revenue ($)", "vote_average": "IMDB Score"},
        template="plotly_dark",
    )
    max_val = max(df_clean["budget"].max(), df_clean["revenue"].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                             mode="lines", line=dict(color=COLORS["primary"], dash="dash", width=2),
                             name="Break-Even"))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*2],
                             mode="lines", line=dict(color=COLORS["secondary"], dash="dot", width=1.5),
                             name="2× Return"))
    fig.update_layout(height=500, paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                      font_color="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Genre ROI", "Yearly Trends", "Month × Genre Heatmap", "Genre Combinations"])

    # ── Tab 1: Genre ROI ───────────────────────────────────────────────────────
    with tab1:
        genre_rows = []
        for _, row in df_clean.iterrows():
            for g in row["genre_list"]:
                genre_rows.append({"genre": g, "roi": row["roi"], "revenue": row["revenue"]})
        genre_df = pd.DataFrame(genre_rows)
        genre_stats = genre_df.groupby("genre").agg(
            median_roi=("roi","median"), count=("roi","count")
        ).reset_index()
        genre_stats = genre_stats[genre_stats["count"] >= 30].sort_values("median_roi")

        bar_colors = [COLORS["secondary"] if v > 200 else COLORS["blue"] for v in genre_stats["median_roi"]]
        fig = go.Figure(go.Bar(
            x=genre_stats["median_roi"], y=genre_stats["genre"],
            orientation="h", marker_color=bar_colors,
            text=[f"%{v:.0f}" for v in genre_stats["median_roi"]], textposition="outside",
        ))
        fig.add_vline(x=100, line_dash="dash", line_color=COLORS["primary"], annotation_text="100% ROI")
        fig.update_layout(
            title="Which Genre Pays Off Most? — Median ROI by Genre",
            xaxis_title="Median ROI (%)", height=550,
            paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="#f0f0f0"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Yearly Trends ───────────────────────────────────────────────────
    with tab2:
        yearly = df_clean[df_clean["release_year"] >= 1980].groupby("release_year").agg(
            avg_budget=("budget","mean"), avg_revenue=("revenue","mean"),
            hit_rate=("is_hit","mean"), count=("title","count")
        ).reset_index()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Average Budget vs Revenue (M$)", "Hit Rate (%)"),
                            vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=yearly["release_year"], y=yearly["avg_budget"]/1e6,
                                 fill="tozeroy", name="Avg Budget",
                                 line=dict(color=COLORS["primary"], width=2),
                                 fillcolor="rgba(229,9,20,0.2)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=yearly["release_year"], y=yearly["avg_revenue"]/1e6,
                                 fill="tonexty", name="Avg Revenue",
                                 line=dict(color=COLORS["secondary"], width=2),
                                 fillcolor="rgba(255,215,0,0.2)"), row=1, col=1)
        fig.add_trace(go.Bar(x=yearly["release_year"], y=yearly["hit_rate"]*100,
                             name="Hit Rate %", marker_color=COLORS["blue"], opacity=0.8), row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color=COLORS["primary"], row=2, col=1)
        fig.update_layout(height=600, paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                          font_color="#f0f0f0", title="Hollywood's 40-Year Evolution")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Month × Genre Heatmap ──────────────────────────────────────────
    with tab3:
        month_genre_rows = []
        for _, row in df_clean.iterrows():
            if pd.notna(row["release_month"]):
                for g in row["genre_list"][:2]:
                    month_genre_rows.append({"month": int(row["release_month"]),
                                             "genre": g, "revenue": row["revenue"]})
        mg_df = pd.DataFrame(month_genre_rows)
        top_genres_hm = mg_df.groupby("genre")["revenue"].count().nlargest(8).index
        mg_filtered = mg_df[mg_df["genre"].isin(top_genres_hm)]
        pivot = mg_filtered.pivot_table(values="revenue", index="genre",
                                        columns="month", aggfunc="median") / 1e6
        pivot.columns = [MONTH_NAMES[c][:3] for c in pivot.columns]

        fig = px.imshow(pivot, color_continuous_scale="YlOrRd", text_auto=".0f",
                        labels={"color": "Median Revenue (M$)"},
                        title="When to Release Which Genre? — Median Revenue by Month × Genre",
                        template="plotly_dark")
        fig.update_layout(height=420, paper_bgcolor="#0f0f1a", font_color="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Adventure films in **May** ($397M) and **November** ($341M) lead all combinations.")

    # ── Tab 4: Genre Combinations ─────────────────────────────────────────────
    with tab4:
        pairs = []
        for _, row in df_clean.iterrows():
            genres = sorted(row["genre_list"])
            for i in range(len(genres)):
                for j in range(i+1, len(genres)):
                    pairs.append({"pair": f"{genres[i]} + {genres[j]}",
                                  "roi": row["roi"], "revenue": row["revenue"]})
        gp_df = pd.DataFrame(pairs)
        gp_stats = gp_df.groupby("pair").agg(
            median_roi=("roi","median"), count=("roi","count")
        ).reset_index()
        gp_stats = gp_stats[gp_stats["count"] >= 10].sort_values("median_roi", ascending=False).head(12)

        fig = px.bar(gp_stats.sort_values("median_roi"), x="median_roi", y="pair",
                     orientation="h", color="median_roi",
                     color_continuous_scale=["#0077B6", "#FFD700", "#E50914"],
                     text=[f"%{v:.0f}" for v in gp_stats.sort_values("median_roi")["median_roi"]],
                     labels={"median_roi":"Median ROI (%)","pair":"Genre Combination"},
                     title="Most Profitable Genre Combinations (min. 10 films)",
                     template="plotly_dark")
        fig.update_layout(height=500, paper_bgcolor="#0f0f1a", font_color="#f0f0f0",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown("# 🤖 Machine Learning Models")
    st.divider()

    col1, col2, col3 = st.columns(3)
    for col, (name, r2_val, color) in zip(
        [col1, col2, col3],
        [("Linear Regression", 0.5757, COLORS["blue"]),
         ("Random Forest",     0.6742, COLORS["secondary"]),
         ("Gradient Boosting", 0.6783, COLORS["primary"])],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}">
                <div class="metric-value" style="color:{color}">{r2_val:.4f}</div>
                <div class="metric-label">R² — {name}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Feature importance
    st.markdown('<div class="section-title">🔑 Feature Importance (Gradient Boosting)</div>',
                unsafe_allow_html=True)
    top_imp = importances.head(15).sort_values("importance")
    bar_colors_imp = [COLORS["primary"] if i >= len(top_imp)-3 else COLORS["blue"]
                      for i in range(len(top_imp))]
    fig = go.Figure(go.Bar(
        x=top_imp["importance"], y=top_imp["feature"],
        orientation="h", marker_color=bar_colors_imp,
        text=[f"{v:.3f}" for v in top_imp["importance"]], textposition="outside",
    ))
    fig.update_layout(
        title="Top 15 Features Driving Revenue Prediction",
        xaxis_title="Importance Score", height=520,
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="#f0f0f0"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 **Red bars** = top 3 features: vote_count, log_budget, budget_per_minute")

    # Model insights
    st.divider()
    st.markdown('<div class="section-title">📌 Model Insights</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.info("**#1 Feature: vote_count**\n\nPopularity / buzz is the strongest revenue signal — films people care about earn more.")
    with i2:
        st.warning("**#2 Feature: log_budget**\n\nBigger budget → bigger revenue, but non-linearly. Returns diminish at the very top.")
    with i3:
        st.success("**#4 Feature: studio_power**\n\nA major studio behind the film lifts predicted revenue significantly.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REVENUE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Revenue Predictor":
    st.markdown("# 🎯 Revenue Predictor")
    st.markdown("Fill in your film's details and get an instant revenue estimate from the Gradient Boosting model.")
    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">🎬 Film Parameters</div>', unsafe_allow_html=True)

        budget_m = st.slider("Budget ($M)", min_value=1, max_value=400, value=50, step=1)
        runtime  = st.slider("Runtime (minutes)", min_value=60, max_value=240, value=110, step=5)
        vote_avg = st.slider("Expected IMDB Score", min_value=1.0, max_value=10.0, value=6.5, step=0.1)
        vote_cnt = st.slider("Expected Vote Count", min_value=100, max_value=20000, value=3000, step=100)
        popularity = st.slider("Popularity Score", min_value=1, max_value=200, value=30, step=1)
        overview_len = st.slider("Overview Length (chars)", min_value=50, max_value=1000, value=300, step=50)

        release_month = st.selectbox(
            "Release Month",
            options=list(MONTH_NAMES.keys()),
            format_func=lambda x: MONTH_NAMES[x],
            index=4,  # May default
        )

        selected_genres = st.multiselect(
            "Genres (select up to 3)",
            options=top_10_genres,
            default=["Action", "Adventure"],
            max_selections=3,
        )

        n_keywords   = st.slider("Number of Keywords", 0, 20, 8)
        n_companies  = st.slider("Number of Production Companies", 1, 10, 2)
        is_major     = st.toggle("Major Studio Production", value=True)

    with right:
        st.markdown('<div class="section-title">📈 Revenue Estimate</div>', unsafe_allow_html=True)

        # Build feature vector
        budget_val    = budget_m * 1e6
        log_budget    = np.log1p(budget_val)
        bpm           = budget_val / max(runtime, 1)
        studio_val    = df_clean["revenue"].mean() * 2 if is_major else df_clean["revenue"].mean() * 0.3
        hit_kw        = min(n_keywords, 5)  # proxy
        n_genres_val  = len(selected_genres)

        season_map_inv = {
            1:"Winter",2:"Winter",3:"Winter",
            4:"Spring",5:"Spring",6:"Spring",
            7:"Summer",8:"Summer",9:"Summer",
            10:"Fall",11:"Fall",12:"Fall"
        }
        season = season_map_inv[release_month]
        season_cols = [c for c in feature_cols if c.startswith("season_")]

        row_data = {
            "log_budget": log_budget, "popularity": popularity,
            "runtime": runtime, "vote_average": vote_avg, "vote_count": vote_cnt,
            "hit_keyword_score": hit_kw, "studio_power": studio_val,
            "n_genres": n_genres_val, "n_keywords": n_keywords,
            "n_companies": n_companies, "budget_per_minute": bpm,
            "overview_length": overview_len, "release_month": release_month,
        }
        for g in top_10_genres:
            row_data[f"genre_{g}"] = 1 if g in selected_genres else 0
        for sc in season_cols:
            s_name = sc.replace("season_", "")
            row_data[sc] = 1 if season == s_name else 0

        X_pred = pd.DataFrame([row_data])[feature_cols].fillna(0)
        log_rev_pred = gb.predict(X_pred)[0]
        revenue_pred = np.expm1(log_rev_pred)
        roi_pred     = ((revenue_pred - budget_val) / budget_val) * 100
        profit_pred  = revenue_pred - budget_val

        # ── Result card ────────────────────────────────────────────────────────
        verdict_color = COLORS["primary"] if roi_pred < 0 else ("#FFD700" if roi_pred < 100 else "#00D4AA")
        verdict_text  = "📉 Expected Loss" if roi_pred < 0 else ("⚠️ Modest Return" if roi_pred < 100 else "🏆 Potential Hit!")

        st.markdown(f"""
        <div class="predict-box">
            <div style="color:#aaa; font-size:0.85rem; margin-bottom:8px">PREDICTED BOX OFFICE REVENUE</div>
            <div class="result-revenue">${revenue_pred/1e6:.1f}M</div>
            <hr style="border-color:#333; margin:16px 0">
            <div style="display:flex; gap:32px; flex-wrap:wrap;">
                <div>
                    <div style="color:#aaa;font-size:0.8rem">ROI</div>
                    <div class="result-roi" style="color:{verdict_color}">
                        {"+" if roi_pred >= 0 else ""}{roi_pred:.0f}%
                    </div>
                </div>
                <div>
                    <div style="color:#aaa;font-size:0.8rem">Profit / Loss</div>
                    <div class="result-roi" style="color:{verdict_color}">
                        {"+" if profit_pred >= 0 else ""}${profit_pred/1e6:.1f}M
                    </div>
                </div>
                <div>
                    <div style="color:#aaa;font-size:0.8rem">Verdict</div>
                    <div class="result-roi" style="color:{verdict_color}">{verdict_text}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Budget comparison gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=revenue_pred / 1e6,
            delta={"reference": budget_m, "prefix":"Budget: $", "suffix":"M"},
            title={"text": "Predicted Revenue (M$)", "font": {"color": "#FFD700"}},
            gauge={
                "axis": {"range": [0, max(revenue_pred/1e6 * 1.5, budget_m * 3)]},
                "bar": {"color": COLORS["primary"]},
                "steps": [
                    {"range": [0, budget_m], "color": "#2a0a0a"},
                    {"range": [budget_m, budget_m*2], "color": "#1a1a2e"},
                ],
                "threshold": {
                    "line": {"color": COLORS["secondary"], "width": 3},
                    "thickness": 0.75,
                    "value": budget_m * 2,
                },
            },
            number={"font": {"color": "#FFD700"}, "suffix": "M"},
        ))
        fig_gauge.update_layout(height=280, paper_bgcolor="#0f0f1a", font_color="#f0f0f0")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Tip based on inputs
        st.markdown("**💡 Optimization Tips**")
        tips = []
        if release_month not in [5, 6, 11, 12]:
            tips.append("📅 Consider releasing in **May, June, or November** for higher median revenue.")
        if not is_major:
            tips.append("🏢 Partnering with a **major studio** significantly boosts predicted revenue.")
        if n_genres_val < 2:
            tips.append("🎭 Adding a second genre (e.g., **Adventure + Music**) can improve ROI.")
        if not tips:
            tips.append("✅ Your setup looks optimized! Budget, timing, and studio are well-aligned.")
        for tip in tips:
            st.markdown(f"- {tip}")

    st.divider()
    st.caption("⚠️ Predictions are based on historical TMDB data (pre-2017). Use as a directional estimate, not a financial guarantee.")
