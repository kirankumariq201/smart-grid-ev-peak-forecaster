# smart-grid-ev-peak-forecaster
Real-time predictive maintenance and overload risk forecasting for municipal EV charging stations using XGBoost, cyclical temporal encoding, and Streamlit.

# ⚡ Smart Grid EV Peak Risk Forecaster
An end-to-end machine learning system engineered to predict regional power grid stress and overload risks driven by Electric Vehicle (EV) charging telemetry. Deployed as an interactive dashboard and mobile-optimized Progressive Web Application (PWA).
---
## 📌 Project Overview
Rapid scaling of electric mobility introduces severe, localized load spikes on municipal electrical distribution feeders. This project processes real-time station metrics (vehicle volume, average dwell duration, aggregate energy output, renewable offsets) alongside temporal cycles to predict grid vulnerability tiers (**Low**, **Medium**, **High**) with **98% classification accuracy**.
---
## 🚀 Key Technical Highlights
* **Physics-Informed Feature Engineering:** Formulated an instantaneous load pressure index combining active station draw against dynamic baseline grid telemetry and renewable absorption buffers.
* **Cyclical Trigonometric Transformations:** Encoded 24-hour diurnal and 7-day weekly periodicities using $\sin(\theta)$ and $\cos(\theta)$ projections to eliminate edge-boundary discontinuities (e.g., transitions between 23:00 and 00:00).
* **Production Modeling Pipeline:** Implemented an integrated Scikit-Learn `ColumnTransformer` (handling one-hot categorical encoding and numerical pass-throughs) coupled with an optimized multi-class **XGBoost Classifier**.
* **Zero-Downtime Edge Deployment:** Packaged using Streamlit Community Cloud and configured for installation as a standalone mobile application via PWA standards.
---
## 📊 Model Evaluation & Performance
The pipeline was validated using stratified 80/20 train-test splits on telemetry records:

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **98.0%** |
| **Macro Average F1-Score** | **0.98** |
| **Weighted Precision** | **0.98** |
| **Weighted Recall** | **0.98** |

---
## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib
* **Interface & Deployment:** Streamlit, Streamlit Community Cloud, PWA
---
## 📂 Repository Structure
```text
├── app.py                  # Streamlit application UI and inference runtime
├── ev_grid_model.pkl       # Serialized XGBoost pipeline and class encoder
├── requirements.txt        # Production dependency specifications
└── README.md               # Technical project documentation