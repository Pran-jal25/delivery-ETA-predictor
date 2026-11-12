import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from feature_engineering import get_feature_columns

os.makedirs("model", exist_ok=True)


def evaluate(name, model, X_test, y_test):
    # Predict and calculate how accurate the model is
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"\n  {name}")
    print(f"    MAE  (average error in minutes) : {mae:.2f}")
    print(f"    RMSE                            : {rmse:.2f}")
    print(f"    R2 Score (1.0 is best)          : {r2:.4f}")
    return {'name': name, 'model': model, 'mae': mae, 'rmse': rmse, 'r2': r2}


def train():

    print("Loading data...")
    df = pd.read_csv("data/featured_data.csv")

    # Get the feature columns we defined in feature_engineering.py
    feature_cols = get_feature_columns()

    # If any column is missing from the data, add it with value 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols]        # input features
    y = df['Time_taken(min)']   # what we want to predict

    # Drop rows that still have missing values
    data = pd.concat([X, y], axis=1).dropna()
    X = data[feature_cols]
    y = data['Time_taken(min)']
    print(f"{len(X)} rows ready for training")

    # Split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training: {len(X_train)} rows | Testing: {len(X_test)} rows")

    # Train 3 models and compare them
    print("\nTraining models...")
    results = []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate("Linear Regression", lr, X_test, y_test))

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                       random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test, y_test))

    # Pick the model with the lowest average error (MAE)
    best = min(results, key=lambda x: x['mae'])
    print(f"\nBest Model: {best['name']}  |  MAE: {best['mae']:.2f} min  |  R2: {best['r2']:.4f}")

    # Save feature importance (which features mattered most)
    best_model = best['model']
    if hasattr(best_model, 'feature_importances_'):
        imp = pd.DataFrame({'Feature': feature_cols, 'Importance': best_model.feature_importances_})
        imp = imp.sort_values('Importance', ascending=False)
        imp.to_csv("model/feature_importance.csv", index=False)
        print("\nTop 5 Important Features:")
        print(imp.head(5).to_string(index=False))

    # Save the model so the app can load it later
    with open("model/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open("model/model_info.pkl", "wb") as f:
        pickle.dump({'name': best['name'], 'mae': round(best['mae'], 2),
                     'rmse': round(best['rmse'], 2), 'r2': round(best['r2'], 4)}, f)

    print("\nModel saved to model/best_model.pkl")
    return best_model, feature_cols


if __name__ == "__main__":
    train()
