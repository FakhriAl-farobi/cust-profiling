# Customer Profiling & Behavioral Analysis

This project includes a platform for customer profiling from transaction data. RFM is used to discover large clusters, while unsupervised machine learning algorithms are used to uncover micro-behaviors as customer subclusters.

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
