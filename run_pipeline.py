import pandas as pd
import os

# Run this file ONCE to clean data and prepare features.
# Then open train_model.ipynb to train the model.
# Then run: streamlit run app.py

print("=" * 50)
print("  DELIVERY ETA PREDICTOR - DATA PIPELINE")
print("=" * 50)

os.makedirs("data",  exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Step 1: Clean the raw data
print("\n[Step 1/2] Cleaning data...")
from data_cleaning import clean_data
df = clean_data()
df.to_csv("data/cleaned_data.csv", index=False)

# Step 2: Add new features
print("\n[Step 2/2] Adding features...")
from feature_engineering import add_features
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df = add_features(df)
df.to_csv("data/featured_data.csv", index=False)

print("\n" + "=" * 50)
print("  DONE!")
print("  Next: open train_model.ipynb to train the model")
print("  Then: streamlit run app.py")
print("=" * 50)
