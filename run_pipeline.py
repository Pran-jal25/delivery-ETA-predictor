import pandas as pd
import os

# Run this file ONCE before launching the app.
# It will: clean data → add features → train model → save charts

print("=" * 50)
print("  DELIVERY ETA PREDICTOR - SETUP PIPELINE")
print("=" * 50)

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Step 1: Clean the raw data
print("\n[Step 1/4] Cleaning data...")
from data_cleaning import clean_data
df = clean_data()
df.to_csv("data/cleaned_data.csv", index=False)

# Step 2: Add new features
print("\n[Step 2/4] Adding features...")
from feature_engineering import add_features
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df = add_features(df)
df.to_csv("data/featured_data.csv", index=False)

# Step 3: Create EDA charts
print("\n[Step 3/4] Creating charts...")
from eda import run_eda
run_eda(df)

# Step 4: Train models and save the best one
print("\n[Step 4/4] Training models...")
from train_model import train
model, features = train()

print("\n" + "=" * 50)
print("  DONE! Now run:  streamlit run app.py")
print("=" * 50)
