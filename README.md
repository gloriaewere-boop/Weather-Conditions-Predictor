# 🌦️ Weather Conditions Predictor (Week 10)

This project predicts simple weather conditions — **Dry**, **Rainy**, or **Snowy** — from a local weather dataset.

It cleans the data, builds time features (year/month/day/dayofweek), trains models (Logistic Regression and Random Forest), and shows results like accuracy, F1-score, and a confusion matrix picture.

---

## 📁 What’s in this project
- `Week10_Weather_Conditions_Predictor_PRECONFIGURED.ipynb` → Main notebook you can run top-to-bottom
- `local_weather.csv` (or `local_weather (1).csv`) → Your dataset
- `figures/` → Pictures like `confusion_matrix.png`
- `artifacts/` → Saved model (`weather_conditions_model.pkl`) and a report (`report.json`)

---

## ▶️ How to run
1. Open the notebook in Jupyter or VS Code.
2. Make sure the dataset file name in the config cell matches your file (e.g., `local_weather (1).csv`).
3. Run the cells from top to bottom.
4. Check the outputs in `figures/` and `artifacts/`.

---

## ✨ What this project does
- Creates a **Condition** label from PRCP (rain) and SNOW columns
- Adds **time features** (year, month, day, dayofweek)
- Preprocesses data (imputes missing values, scales numbers, one-hot encodes categories)
- Trains models and **tunes** the best one
- Saves a trained **model** and a short **report**

---

## 🧪 Results (example items you will see)
- `report.json` → shows accuracy and best model parameters
- `figures/confusion_matrix.png` → a grid showing how well predictions match the truth

---

## 📝 Author
- **Name:** Gloria Ighagbon
- **Email:** gloriaewere@gmail.com
