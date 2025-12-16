# Predictive Public Safety Resource Allocation

**A Machine Learning pipeline to optimize police patrol deployment using spatial-temporal forecasting.**

![Status](https://img.shields.io/badge/Status-Complete-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This project analyzes **Las Vegas Metropolitan Police Department (LVMPD)** calls for service data to predict high-risk "hot spots" for specific time windows. By training a **Random Forest Regressor** on historical 2023 data and validating it against 2024 data, the system recommends optimal patrol zones to reduce response times and improve public safety resource allocation.

**Key Objective:** Move beyond static "heat maps" to dynamic, predictive resource allocation that accounts for day-of-week and time-of-day variations.

## 📊 Key Results
* **Model Accuracy:** Achieved an **R² of 0.82**, explaining 82% of the variance in crime density.
* **Precision:** Mean Absolute Error (**MAE**) of **3.26**, meaning predictions are accurate within ~3 incidents per zone.
* **Temporal Insight:** Identified that weekend late-night shifts require **3.25x** more resources than weekday shifts in specific entertainment corridors.

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Geospatial Analysis:** `GeoPandas`, `H3pandas` (Uber's Hexagonal Hierarchical Spatial Index), `Contextily`
* **Machine Learning:** `Scikit-Learn` (Random Forest Regressor)
* **Visualization:** `Seaborn`, `Matplotlib`
* **Data Processing:** `Pandas`, `Parquet` for efficient storage.

### The Pipeline
1.  **Ingestion & Cleaning:** Parsed 200,000+ GeoJSON records, fixed "Midnight Spike" artifacts, and engineered features for time-of-day buckets.
2.  **Spatial Indexing:** Aggregated individual points into **H3 Hexagons (Resolution 8)** to normalize high-density areas.
3.  **Training:** Trained a Random Forest model on 2023 data (`train_2023.csv`).
4.  **Validation:** Tested predictions against unseen 2024 data (`test_2024.csv`) to ensure temporal stability.
5.  **Deployment:** Built a recommendation engine that generates patrol maps for specific user queries (e.g., "Saturday Late Night").

## 📂 Project Structure
```text
├── data/
│   ├── cleaned_lvmpd_2023.parquet   # Processed 2023 data
│   ├── cleaned_lvmpd_2024.parquet   # Processed 2024 data
│   └── ml_ready/                    # H3 Aggregated Training Data
├── reports/figures/                 # Generated Maps & Charts
├── data_cleaning.py                 # Cleaning & Feature Engineering
├── eda.py                           # Exploratory Data Analysis
├── ml_prep.py                       # H3 Hexagon Grid Generation
├── train_model.py                   # Model Training & Evaluation
├── recommendation_engine.py         # Deployment Tool (Generates Maps)
└── error_analysis.py                # Residual Analysis (Where did the model fail?)