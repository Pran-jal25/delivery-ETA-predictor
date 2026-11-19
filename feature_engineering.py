import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

df = pd.read_csv("data/cleaned_data.csv")
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

# Distance between restaurant and delivery location (Haversine formula)
def haversine(row):
    lat1, lon1, lat2, lon2 = map(radians, [
        row['Restaurant_latitude'], row['Restaurant_longitude'],
        row['Delivery_location_latitude'], row['Delivery_location_longitude']
    ])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1-a))

df['distance_km'] = df.apply(haversine, axis=1)

# Pickup delay: minutes between order placed and picked up
def to_min(t):
    try:
        h, m = str(t).split(':')[:2]
        return int(h) * 60 + int(m)
    except:
        return np.nan

df['order_time_min']   = df['Time_Orderd'].apply(to_min)
df['pickup_time_min']  = df['Time_Order_picked'].apply(to_min)
df['pickup_delay_min'] = (df['pickup_time_min'] - df['order_time_min']).apply(
    lambda x: x + 1440 if x < 0 else x
)

# Hour the order was placed
df['order_hour'] = df['order_time_min'].apply(
    lambda x: int(x // 60) if not np.isnan(x) else np.nan
)

# Is it a weekend?
df['is_weekend'] = df['Order_Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

# Encode categories as numbers
df['traffic_encoded']    = df['Road_traffic_density'].map({'Low':1,'Medium':2,'High':3,'Jam':4}).fillna(2)
df['city_encoded']       = df['City'].map({'Urban':1,'Metropolitian':2,'Semi-Urban':3}).fillna(1)
df['vehicle_encoded']    = df['Type_of_vehicle'].map({'bicycle':1,'scooter':2,'electric_scooter':3,'motorcycle':4}).fillna(2)
df['order_type_encoded'] = df['Type_of_order'].map({'Snack':1,'Drinks':2,'Buffet':3,'Meal':4}).fillna(2)
df['is_festival']        = df['Festival'].map({'Yes':1,'No':0}).fillna(0)

# Weather one-hot columns (no NaN column)
weather_dummies = pd.get_dummies(df['Weatherconditions'], prefix='weather', dummy_na=False)
df = pd.concat([df, weather_dummies], axis=1)

print(f"Features added. Dataset now has {df.shape[1]} columns.")
df.to_csv("data/featured_data.csv", index=False)
print("Saved to data/featured_data.csv")

# Feature columns used by the model
FEATURE_COLS = [
    'Delivery_person_Age', 'Delivery_person_Ratings',
    'distance_km', 'pickup_delay_min', 'order_hour',
    'traffic_encoded', 'city_encoded', 'vehicle_encoded',
    'order_type_encoded', 'multiple_deliveries',
    'is_festival', 'is_weekend', 'Vehicle_condition',
    'weather_Cloudy', 'weather_Fog', 'weather_Sandstorms',
    'weather_Stormy', 'weather_Sunny', 'weather_Windy'
]
