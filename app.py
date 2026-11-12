import streamlit as st
import pandas as pd
import pickle
import os
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="Delivery ETA Predictor", page_icon="🚚")

# ------------------------------------------------------------------
# All 22 cities from the dataset with their areas and city type
# City type (Urban/Metropolitian/Semi-Urban) is auto-assigned
# ------------------------------------------------------------------
AREAS = {
    "Jaipur":      {"type": "Metropolitian", "areas": {"Pink City":(26.9124,75.7873), "Malviya Nagar":(26.8505,75.8069), "Vaishali Nagar":(26.9100,75.7369), "Mansarovar":(26.8504,75.7577), "C-Scheme":(26.9057,75.8023)}},
    "Ranchi":      {"type": "Urban",         "areas": {"Lalpur":(23.3441,85.3096), "Doranda":(23.3315,85.3097), "Kanke Road":(23.3850,85.3250), "Harmu":(23.3608,85.3001), "Ratu Road":(23.3980,85.3240)}},
    "Bangalore":   {"type": "Metropolitian", "areas": {"Koramangala":(12.9279,77.6271), "Indiranagar":(12.9784,77.6408), "Whitefield":(12.9698,77.7500), "MG Road":(12.9757,77.6011), "Electronic City":(12.8399,77.6770)}},
    "Surat":       {"type": "Metropolitian", "areas": {"Adajan":(21.2060,72.8311), "Vesu":(21.1498,72.7984), "Piplod":(21.1500,72.8167), "Athwa":(21.1941,72.8311), "Katargam":(21.2305,72.8398)}},
    "Hyderabad":   {"type": "Metropolitian", "areas": {"Banjara Hills":(17.4156,78.4347), "Hitech City":(17.4435,78.3772), "Gachibowli":(17.4401,78.3489), "Madhapur":(17.4418,78.3912), "Secunderabad":(17.4399,78.4983)}},
    "Mumbai":      {"type": "Metropolitian", "areas": {"Andheri":(19.1136,72.8697), "Bandra":(19.0596,72.8295), "Powai":(19.1176,72.9060), "Dadar":(19.0178,72.8478), "Juhu":(19.1075,72.8263)}},
    "Mysore":      {"type": "Urban",         "areas": {"Vijayanagar":(12.3218,76.6107), "Kuvempunagar":(12.2958,76.6394), "Hebbal":(12.3613,76.6220), "Gokulam":(12.3301,76.6192), "Nazarbad":(12.3024,76.6376)}},
    "Coimbatore":  {"type": "Metropolitian", "areas": {"RS Puram":(11.0168,76.9558), "Gandhipuram":(11.0183,76.9725), "Peelamedu":(11.0280,77.0280), "Saibaba Colony":(11.0105,76.9437), "Singanallur":(10.9939,77.0197)}},
    "Vadodara":    {"type": "Metropolitian", "areas": {"Alkapuri":(22.3119,73.1723), "Fatehgunj":(22.3217,73.1851), "Manjalpur":(22.2678,73.1759), "Gotri":(22.3350,73.1467), "Waghodia Road":(22.2850,73.2167)}},
    "Indore":      {"type": "Metropolitian", "areas": {"Vijay Nagar":(22.7533,75.8937), "Palasia":(22.7196,75.8577), "Scheme 54":(22.7400,75.8800), "Bhawarkua":(22.6882,75.8694), "Sudama Nagar":(22.7050,75.8750)}},
    "Chennai":     {"type": "Metropolitian", "areas": {"Anna Nagar":(13.0850,80.2101), "T Nagar":(13.0418,80.2341), "Velachery":(12.9815,80.2180), "Adyar":(13.0012,80.2565), "Porur":(13.0346,80.1572)}},
    "Pune":        {"type": "Metropolitian", "areas": {"Koregaon Park":(18.5362,73.8938), "Hinjewadi":(18.5912,73.7389), "Baner":(18.5590,73.7868), "Kothrud":(18.5074,73.8077), "Viman Nagar":(18.5679,73.9143)}},
    "Agra":        {"type": "Urban",         "areas": {"Taj Ganj":(27.1724,78.0421), "Sikandra":(27.2097,77.9850), "Kamla Nagar":(27.1900,78.0100), "Shahganj":(27.1800,78.0200), "Bodla":(27.1500,78.0600)}},
    "Ludhiana":    {"type": "Metropolitian", "areas": {"Model Town":(30.9100,75.8500), "Civil Lines":(30.9010,75.8600), "Sarabha Nagar":(30.8950,75.8450), "BRS Nagar":(30.8800,75.8200), "Dugri":(30.8700,75.8600)}},
    "Allahabad":   {"type": "Urban",         "areas": {"Civil Lines":(25.4500,81.8500), "Naini":(25.3800,81.8700), "George Town":(25.4358,81.8463), "Lukerganj":(25.4600,81.8200), "Prayagraj":(25.4484,81.8322)}},
    "Kanpur":      {"type": "Metropolitian", "areas": {"Swaroop Nagar":(26.4800,80.3300), "Civil Lines":(26.4700,80.3500), "Kidwai Nagar":(26.4700,80.3200), "Kalyanpur":(26.5000,80.2900), "Armapur":(26.5050,80.3000)}},
    "Dehradun":    {"type": "Urban",         "areas": {"Rajpur Road":(30.3500,78.0700), "Patel Nagar":(30.3200,78.0400), "Race Course":(30.3300,77.9900), "Clement Town":(30.2900,77.9800), "Sahastradhara":(30.3800,78.1000)}},
    "Goa":         {"type": "Semi-Urban",    "areas": {"Panaji":(15.4909,73.8278), "Mapusa":(15.5937,73.8117), "Margao":(15.2832,73.9862), "Vasco":(15.3980,73.8170), "Calangute":(15.5438,73.7553)}},
    "Aurangabad":  {"type": "Urban",         "areas": {"Cidco":(19.8762,75.3433), "Garkheda":(19.8200,75.3500), "Waluj":(19.8100,75.2500), "Osmanpura":(19.8800,75.3200), "Cantonment":(19.8900,75.3400)}},
    "Kochi":       {"type": "Metropolitian", "areas": {"Ernakulam":(9.9816,76.2999), "Kakkanad":(10.0167,76.3500), "Edapally":(10.0200,76.3100), "Fort Kochi":(9.9633,76.2433), "Aluva":(10.1070,76.3570)}},
    "Kolkata":     {"type": "Metropolitian", "areas": {"Park Street":(22.5513,88.3514), "Salt Lake":(22.5700,88.4100), "Howrah":(22.5958,88.2636), "Dum Dum":(22.6500,88.4200), "Behala":(22.5000,88.3200)}},
    "Bhopal":      {"type": "Metropolitian", "areas": {"MP Nagar":(23.2332,77.4336), "Arera Colony":(23.2050,77.4350), "Kolar Road":(23.1700,77.4700), "Govindpura":(23.2500,77.4700), "Shahpura":(23.1900,77.4400)}},
}


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


@st.cache_resource
def load_model():
    with open("model/best_model.pkl",      "rb") as f: model    = pickle.load(f)
    with open("model/feature_columns.pkl", "rb") as f: features = pickle.load(f)
    with open("model/model_info.pkl",      "rb") as f: info     = pickle.load(f)
    return model, features, info


def build_input(data, feature_cols):
    row = {col: 0 for col in feature_cols}
    # Defaults below are real medians/modes from the training dataset
    row['Delivery_person_Age']     = 30        # median age in dataset
    row['Delivery_person_Ratings'] = 4.7       # median rating in dataset
    row['distance_km']             = data['distance']
    row['pickup_delay_min']        = 10        # median prep time in dataset
    row['order_hour']              = data['hour']
    row['traffic_encoded']         = {'Low':1,'Medium':2,'High':3,'Jam':4}[data['traffic']]
    row['city_encoded']            = {'Urban':1,'Metropolitian':2,'Semi-Urban':3}[data['city_type']]
    row['vehicle_encoded']         = 4         # mode: motorcycle (most common in dataset)
    row['order_type_encoded']      = {'Snack':1,'Drinks':2,'Buffet':3,'Meal':4}[data['order_type']]
    row['multiple_deliveries']     = 1         # median in dataset
    row['is_festival']             = 1 if data['festival'] == 'Yes' else 0
    row['is_weekend']              = data['weekend']
    row['Vehicle_condition']       = 1         # median in dataset (1 = Good)
    if f"weather_{data['weather']}" in row:
        row[f"weather_{data['weather']}"] = 1
    return pd.DataFrame([row])[feature_cols]


# ── UI ──────────────────────────────────────────────────────────

st.title("🚚 Delivery ETA Predictor")
st.write("Fill in your order details to get the estimated delivery time.")
st.markdown("---")

if not os.path.exists("model/best_model.pkl"):
    st.error("Model not found. Please run `python run_pipeline.py` first to train the model.")
    st.stop()

model, feature_cols, info = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location")
    city          = st.selectbox("Select City", list(AREAS.keys()))
    areas         = list(AREAS[city]["areas"].keys())
    restaurant_area = st.selectbox("Restaurant Area",    areas, index=0)
    delivery_area   = st.selectbox("Your Delivery Area", areas, index=1)

    r_lat, r_lon  = AREAS[city]["areas"][restaurant_area]
    d_lat, d_lon  = AREAS[city]["areas"][delivery_area]
    dist          = calculate_distance(r_lat, r_lon, d_lat, d_lon)
    city_type     = AREAS[city]["type"]   # auto-assigned, customer doesn't see this

    st.metric("Estimated Distance", f"{dist:.2f} km")

    st.subheader("🍔 Order")
    order_type = st.selectbox("What are you ordering?", ["Snack", "Drinks", "Meal", "Buffet"])
    hour       = st.slider("Order Time (Hour of Day)", 0, 23, 19, format="%d:00")

with col2:
    st.subheader("🌦️ Current Conditions")
    weather  = st.selectbox("Weather",          ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"])
    traffic  = st.selectbox("Traffic Level",    ["Low", "Medium", "High", "Jam"])
    festival = st.selectbox("Festival Day?",    ["No", "Yes"])
    weekend  = 1 if st.checkbox("Is it a Weekend?") else 0

st.markdown("---")

if st.button("🔮 Predict Delivery Time", use_container_width=True, type="primary"):

    if restaurant_area == delivery_area:
        st.warning("Please select different areas for restaurant and delivery.")
    else:
        input_df   = build_input({
            'distance': dist, 'hour': hour, 'traffic': traffic,
            'city_type': city_type, 'order_type': order_type,
            'festival': festival, 'weekend': weekend, 'weather': weather
        }, feature_cols)

        prediction = int(max(5, round(model.predict(input_df)[0])))

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("⏱️ Estimated Delivery Time", f"{prediction} minutes")
        c2.metric("📍 Route", f"{restaurant_area} → {delivery_area}")
        c3.metric("📏 Distance", f"{dist:.2f} km")

        if prediction <= 20:
            st.success(f"🟢 Fast delivery! Your order arrives in about **{prediction} minutes**.")
        elif prediction <= 35:
            st.info(f"🟡 Your order will arrive in about **{prediction} minutes**.")
        else:
            st.warning(f"🔴 Longer than usual — expect about **{prediction} minutes** due to current conditions.")
