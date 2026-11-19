import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("dataset.csv")
print(f"Loaded {df.shape[0]} rows")

# Remove extra spaces
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Fix delivery time: "(min) 24" → 24
df['Time_taken(min)'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)

# Fix weather: "conditions Sunny" → "Sunny"
df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '', regex=False)

# Convert to numbers
df['Delivery_person_Age']     = pd.to_numeric(df['Delivery_person_Age'],     errors='coerce')
df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
df['multiple_deliveries']     = pd.to_numeric(df['multiple_deliveries'],     errors='coerce')

# Fill missing values with median
df['Delivery_person_Age']     = df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median())
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median())
df['multiple_deliveries']     = df['multiple_deliveries'].fillna(df['multiple_deliveries'].median())
df['Time_Orderd']             = df['Time_Orderd'].fillna(df['Time_Orderd'].mode()[0])

# Remove rows with wrong GPS (0, 0)
bad_gps = (df['Restaurant_latitude'] == 0.0) | (df['Restaurant_longitude'] == 0.0)
df = df[~bad_gps].reset_index(drop=True)

# Fix date format
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')

print(f"Cleaning done. {df.shape[0]} rows remaining.")
df.to_csv("data/cleaned_data.csv", index=False)
print("Saved to data/cleaned_data.csv")
