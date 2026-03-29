# 🏠 California House Price Prediction

End-to-end machine learning project to predict median house values using the California Housing dataset.  
Includes data analysis, feature engineering, model selection, and deployment-ready inference pipeline.

---

## 📌 Project Overview

This project builds a regression model to predict housing prices based on demographic and geographic features.  
The focus is on creating a **robust ML pipeline** with proper experimentation and evaluation.

- Designed with a focus on real-world ML practices including reproducibility, modularity, and deployment.
---

## 🌐 Live Demo
👉 [Click here to try the app](https://california-house-price-predictor-kpg9psfdoj9vspwqjbggfk.streamlit.app/)

---

## ⚡ How it Works

User inputs housing features → Data is preprocessed → Model predicts house price in real-time via Streamlit app

---

## 🚀 Best Model

- **Model**: HistGradientBoostingRegressor  
- **R² Score**: 0.844  
- **RMSE**: 45219.895 

---

## 🔄 Project Workflow

1. Exploratory Data Analysis (EDA)  
2. Data Preprocessing & Feature Engineering (ratio features, clustering)  
3. Baseline Model Building  
4. Model Selection using Cross-Validation  
5. Hyperparameter Tuning (GridSearchCV)  
6. Final Model Evaluation  
7. Inference Pipeline for Predictions  

---

## 📁 Project Structure

```bash
california-house-price-predictor/
├── app/
│   ├── app.py
│   └── final_model.joblib
├── notebooks/
│   └── notebook.ipynb
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 📊 Key Decisions & Findings

| Decision | Outcome |
|---|---|
| Dropped $500k capped rows | R² dropped 0.844 → 0.824, reverted |
| Added ratio features | R² improved 0.830 → 0.844 |
| KMeans location clustering | Improved linear models, minimal impact on tree-based models |
| Log transform on target | No improvement, reverted |

---

## 🛠️ Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib
- seaborn
- XGBoost     (used for model comparison)
- Streamlit       
- joblib          

---

## 📊 Dataset

- Source: California Housing Dataset  
**Kaggle link:** https://www.kaggle.com/datasets/camnugent/california-housing-prices

---

## 🎯 Key Highlights

- Built a complete **end-to-end ML pipeline**
- Applied **feature engineering (ratio features, clustering)**
- Compared multiple models using **cross-validation**
- Performed **hyperparameter tuning**
- Designed a reusable **inference pipeline**
- Deployed with **Streamlit**

---
