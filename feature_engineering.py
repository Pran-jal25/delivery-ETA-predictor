import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula — calculates real distance between two GPS points
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def time_to_minutes(t):
    # Converts "HH:MM:SS" to total minutes (e.g. "01:30:00" → 90)
    try:
        parts = str(t).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return np.nan


def add_features(df):

    # Distance between restaurant and delivery location
    df['distance_km'] = df.apply(
        lambda row: calculate_distance(
            row['Restaurant_latitude'], row['Restaurant_longitude'],
            row['Delivery_location_latitude'], row['Delivery_location_longitude']
        ), axis=1
    )

    # How many minutes passed between order placed and order picked up
    df['order_time_min']  = df['Time_Orderd'].apply(time_to_minutes)
    df['pickup_time_min'] = df['Time_Order_picked'].apply(time_to_minutes)
    df['pickup_delay_min'] = df['pickup_time_min'] - df['order_time_min']
    df['pickup_delay_min'] = df['pickup_delay_min'].apply(lambda x: x + 1440 if x < 0 else x)

    # What hour the order was placed (0 to 23)
    df['order_hour'] = df['order_time_min'].apply(lambda x: int(x // 60) if not np.isnan(x) else np.nan)

    # Is it a weekend? (5=Saturday, 6=Sunday)
    df['is_weekend'] = df['Order_Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # Traffic: convert text to number (Low=1, Medium=2, High=3, Jam=4)
    traffic_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}
    df['traffic_encoded'] = df['Road_traffic_density'].map(traffic_map).fillna(2)

    # City: Urban=1, Metropolitian=2, Semi-Urban=3
    df['city_encoded'] = df['City'].map({'Urban': 1, 'Metropolitian': 2, 'Semi-Urban': 3}).fillna(1)

    # Vehicle type: convert text to number
    df['vehicle_encoded'] = df['Type_of_vehicle'].map(
        {'bicycle': 1, 'scooter': 2, 'electric_scooter': 3, 'motorcycle': 4}
    ).fillna(2)

    # Order type: convert text to number
    df['order_type_encoded'] = df['Type_of_order'].map(
        {'Snack': 1, 'Drinks': 2, 'Buffet': 3, 'Meal': 4}
    ).fillna(2)

    # Festival: Yes=1, No=0
    df['is_festival'] = df['Festival'].map({'Yes': 1, 'No': 0}).fillna(0)

    # Weather: create one column per weather type (1 if that weather, else 0)
    # dropna=True ensures rows with missing weather don't create a 'weather_NaN' column
    weather_dummies = pd.get_dummies(df['Weatherconditions'], prefix='weather', dummy_na=False)
    df = pd.concat([df, weather_dummies], axis=1)

    print(f"Features added. Dataset now has {df.shape[1]} columns.")
    return df


def get_feature_columns():
    # This list tells the model which columns to use for training
    base = [
        'Delivery_person_Age', 'Delivery_person_Ratings',
        'distance_km', 'pickup_delay_min', 'order_hour',
        'traffic_encoded', 'city_encoded', 'vehicle_encoded',
        'order_type_encoded', 'multiple_deliveries',
        'is_festival', 'is_weekend', 'Vehicle_condition',
    ]
    weather_cols = ['weather_Cloudy', 'weather_Fog', 'weather_Sandstorms',
                    'weather_Stormy', 'weather_Sunny', 'weather_Windy']
    return base + weather_cols


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df = add_features(df)
    df.to_csv("data/featured_data.csv", index=False)
    print("Saved to data/featured_data.csv")
