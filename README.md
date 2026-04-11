# Customer Profiling & Behavioral Analysis

End-to-end analytical pipeline transforming raw transaction data into actionable business intelligence. This project covers the complete data lifecycle: from data cleaning and pipeline architecture, to advanced feature engineering (RFM + behavioral metrics), unsupervised machine learning, and deployment via an interactive web dashboard.

## Key Features

* **Robust Data Processing:** Extracted and cleaned massive transaction datasets (scaling up to millions of records) utilizing SQL (PostgreSQL) and Pandas to ensure high data integrity.
* **Feature Engineering:** Developed comprehensive customer profiles by calculating Recency, Frequency, and Monetary (RFM) values alongside specific behavioral metrics.
* **Machine Learning Clustering:** Implemented and evaluated unsupervised learning algorithms, specifically K-Means and DBSCAN, to segment users into distinct, actionable behavioral cohorts.
* **Interactive Dashboard:** Deployed a dynamic, web-based Streamlit application featuring comprehensive data visualizations, allowing stakeholders to independently explore cluster characteristics and business insights.

## Tech Stack

* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-learn
* **Database:** PostgreSQL / SQL
* **Web Framework:** Streamlit
* **Visualization:** Matplotlib, Seaborn, Plotly
