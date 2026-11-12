# 🚚 Logistics & Delivery ETA Predictor

A machine learning project that predicts food delivery time (in minutes) based on real-world conditions like weather, traffic, distance, and order details.

Built on a dataset of **45,000+ delivery orders** across 22 Indian cities.

🔗 **Live App:** *(add your Streamlit Cloud link here after deployment)*

---

## 🎯 Results

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | 4.97 min | 0.56 |
| Random Forest | 3.10 min | 0.83 |
| **XGBoost (Best)** | **3.04 min** | **0.84** |

> The model predicts delivery time with an average error of just **±3 minutes**.

---

## 🧠 Top Factors That Affect Delivery Time

1. Traffic density
2. Weather conditions
3. Multiple deliveries
4. Festival day
5. Delivery person rating

---

## 🚀 How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the model (run once)**
```bash
python run_pipeline.py
```

**3. Launch the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── dataset.csv              ← Raw dataset (45,000+ orders)
├── data_cleaning.py         ← Fixes messy data (NaNs, wrong formats)
├── feature_engineering.py   ← Creates new features (distance, time, encoding)
├── eda.py                   ← Generates 10 insight charts
├── train_model.py           ← Trains 3 models, saves the best one
├── run_pipeline.py          ← One command to run everything
├── app.py                   ← Streamlit web app
└── requirements.txt
```

---

## 🛠 Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `Streamlit` · `matplotlib` · `seaborn`
