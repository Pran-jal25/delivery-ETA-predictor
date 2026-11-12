import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")


def run_eda(df):

    print("Creating charts...")

    # Chart 1: How are delivery times spread out?
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Time_taken(min)'], bins=30, kde=True, color='steelblue')
    plt.title('Distribution of Delivery Times')
    plt.xlabel('Time (minutes)')
    plt.tight_layout()
    plt.savefig('plots/01_delivery_time_distribution.png')
    plt.close()

    # Chart 2: Does traffic affect delivery time?
    plt.figure(figsize=(10, 5))
    order = ['Low', 'Medium', 'High', 'Jam']
    sns.boxplot(data=df[df['Road_traffic_density'].isin(order)],
                x='Road_traffic_density', y='Time_taken(min)',
                order=order, hue='Road_traffic_density', palette='Oranges', legend=False)
    plt.title('Delivery Time by Traffic Level')
    plt.xlabel('Traffic')
    plt.tight_layout()
    plt.savefig('plots/02_time_vs_traffic.png')
    plt.close()

    # Chart 3: Does weather affect delivery time?
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x='Weatherconditions', y='Time_taken(min)',
                hue='Weatherconditions', palette='Blues', legend=False)
    plt.title('Delivery Time by Weather')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('plots/03_time_vs_weather.png')
    plt.close()

    # Chart 4: Does city type affect delivery time?
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='City', y='Time_taken(min)',
                hue='City', palette='Greens', legend=False)
    plt.title('Delivery Time by City Type')
    plt.tight_layout()
    plt.savefig('plots/04_time_vs_city.png')
    plt.close()

    # Chart 5: Does distance affect delivery time?
    plt.figure(figsize=(10, 5))
    sample = df.sample(min(3000, len(df)), random_state=42)
    plt.scatter(sample['distance_km'], sample['Time_taken(min)'], alpha=0.3, color='coral')
    plt.title('Delivery Time vs Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Time (minutes)')
    plt.tight_layout()
    plt.savefig('plots/05_time_vs_distance.png')
    plt.close()

    # Chart 6: Which hours have the longest deliveries?
    hourly = df.groupby('order_hour')['Time_taken(min)'].mean().reset_index()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=hourly, x='order_hour', y='Time_taken(min)', marker='o', color='purple')
    plt.title('Average Delivery Time by Hour of Day')
    plt.xlabel('Hour (0-23)')
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig('plots/06_time_vs_hour.png')
    plt.close()

    # Chart 7: Do better-rated riders deliver faster?
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Delivery_person_Ratings', y='Time_taken(min)',
                hue='Delivery_person_Ratings', palette='coolwarm', legend=False)
    plt.title('Delivery Time vs Rider Rating')
    plt.tight_layout()
    plt.savefig('plots/07_time_vs_rating.png')
    plt.close()

    # Chart 8: Which features are most related to delivery time?
    num_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km',
                'pickup_delay_min', 'order_hour', 'traffic_encoded',
                'multiple_deliveries', 'Vehicle_condition', 'Time_taken(min)']
    available = [c for c in num_cols if c in df.columns]
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[available].corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                mask=np.triu(np.ones((len(available), len(available)), dtype=bool)))
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/08_correlation_heatmap.png')
    plt.close()

    # Chart 9: Festival vs normal day
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='Festival', y='Time_taken(min)',
                hue='Festival', palette='Set2', legend=False)
    plt.title('Delivery Time: Festival vs Normal Day')
    plt.tight_layout()
    plt.savefig('plots/09_time_vs_festival.png')
    plt.close()

    # Chart 10: Which vehicle is fastest?
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Type_of_vehicle', y='Time_taken(min)',
                hue='Type_of_vehicle', palette='Set2', legend=False)
    plt.title('Delivery Time by Vehicle Type')
    plt.tight_layout()
    plt.savefig('plots/10_time_vs_vehicle.png')
    plt.close()

    print("All 10 charts saved in the plots/ folder.")


if __name__ == "__main__":
    df = pd.read_csv("data/featured_data.csv")
    run_eda(df)
    print("\nDelivery time stats:")
    print(df['Time_taken(min)'].describe())
