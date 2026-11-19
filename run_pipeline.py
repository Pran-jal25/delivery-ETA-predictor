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

print("\n[Step 1/2] Cleaning data...")
exec(open("data_cleaning.py").read())

print("\n[Step 2/2] Adding features...")
exec(open("feature_engineering.py").read())

print("\n" + "=" * 50)
print("  DONE!")
print("  Next: open train_model.ipynb to train the model")
print("  Then: streamlit run app.py")
print("=" * 50)
