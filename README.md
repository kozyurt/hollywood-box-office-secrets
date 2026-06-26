# 🎬 Hollywood Box Office Secrets
### What Makes a Movie a Hit? — Data Science Analysis on TMDB 5000 Movies

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project analyzes **5,000 Hollywood films** from the TMDB dataset to uncover the patterns behind box office success. Using exploratory data analysis, feature engineering, and machine learning, we build a predictive model for movie revenue and derive actionable insights for filmmakers and studios.

**Best model achieved: Gradient Boosting with R² = 0.6783**

---

## 🔍 Key Findings

| Insight | Detail |
|---|---|
| 🏆 Most Profitable Genre | **Music** (Median ROI: 212%) |
| 📅 Best Release Month | **May & June** (Summer blockbuster window) |
| 💰 Budget Sweet Spot | **< $5M** and **$100M+** have the highest hit rates (~70%) |
| 🤖 Top Predictive Feature | `vote_count` (popularity proxy) |
| 🎭 Best Genre Combo | **Family-Music** (Median ROI: ~265%) |

---

## 📊 Visualizations

| # | Chart | Description |
|---|---|---|
| 01 | Budget vs Revenue | Scatter plot with IMDB score coloring and break-even lines |
| 02 | Genre ROI | Median ROI by genre — which genres pay off most |
| 03 | Yearly Trends | Hollywood's 40-year budget/revenue evolution + hit rate |
| 04 | Month × Genre Heatmap | Optimal release timing by genre |
| 05 | Keyword Cloud | Most used keywords in successful films |
| 06 | Model Comparison | Linear Regression vs Random Forest vs Gradient Boosting |
| 07 | Feature Importance | Top 15 features driving revenue predictions |
| 08 | Film Recipe | Genre combos + release months + budget segments |

---

## 🗂️ Project Structure

```
hollywood-box-office-secrets/
│
├── gise_rekortmen_film_sırları.ipynb   # Main analysis notebook
├── tmdb_5000_movies.csv                # Dataset (TMDB 5000)
├── requirements.txt                    # Python dependencies
├── app.py                              # Streamlit dashboard (coming soon)
│
└── charts/                             # Generated visualizations
    ├── 01_budget_vs_revenue.png
    ├── 02_genre_roi.png
    ├── 03_yearly_trends.png
    ├── 04_month_genre_heatmap.png
    ├── 05_keyword_cloud.png
    ├── 06_model_comparison.png
    ├── 07_feature_importance.png
    └── 08_movie_recipe.png
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hollywood-box-office-secrets.git
cd hollywood-box-office-secrets
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

```bash
jupyter notebook gise_rekortmen_film_sırları.ipynb
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 🤖 Machine Learning Pipeline

### Features Used (28 total)

- **Budget features:** `log_budget`, `budget_per_minute`
- **Engagement:** `popularity`, `vote_average`, `vote_count`
- **Film metadata:** `runtime`, `overview_length`, `release_month`
- **Genre flags:** One-hot encoded top 10 genres
- **Engineered:** `studio_power`, `hit_keyword_score`, `n_genres`, `n_keywords`, `n_companies`
- **Season dummies:** Spring / Summer / Fall / Winter

### Model Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.5757 | — | — |
| Random Forest | 0.6742 | — | — |
| **Gradient Boosting** | **0.6783** | — | — |

Cross-validation R² (Gradient Boosting, 5-fold): **~0.67 ± ~0.04**

---

## 📦 Dataset

**Source:** [TMDB 5000 Movie Dataset — Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

- 4,803 movies after cleaning
- Features: budget, revenue, genres, keywords, production companies, release date, cast, crew, ratings
- Cleaned by removing entries with budget or revenue ≤ $1,000

---

## 🛠️ Tech Stack

- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, wordcloud
- **Machine Learning:** scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor)
- **Dashboard:** Streamlit, Plotly
- **Environment:** Jupyter Notebook, Python 3.10+

---

## 🙋 Author

Made with ❤️ and data curiosity.  
Feel free to open issues, fork, or star ⭐ the repo if you found it useful!
