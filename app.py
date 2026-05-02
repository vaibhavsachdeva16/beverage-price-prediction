import streamlit as st
import pandas as pd
import pickle

# Load model + encoder
model = pickle.load(open("pipeline.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="CodeX Beverage", layout="wide")

# Title
st.markdown(
    "<h1 style='text-align: center; color: white; background-color:#4a73c9; padding:10px;'>CodeX Beverage: Price Prediction</h1>",
    unsafe_allow_html=True
)

# Layout (4 columns)
col1, col2, col3, col4 = st.columns(4)

# -------- COLUMN 1 --------
with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=25)
    income = st.selectbox("Income Level (in L)", ['<10L', '10L - 15L', '16L - 25L', '26L - 35L', '> 35L', 'Not Reported'])
    awareness = st.selectbox("Awareness of other brands", ['0 to 1', '2 to 4', 'above 4'])
    packaging = st.selectbox("Packaging Preference", ['Simple', 'Premium', 'Eco-Friendly'])

# -------- COLUMN 2 --------
with col2:
    gender = st.selectbox("Gender", ['M', 'F'])
    frequency = st.selectbox("Consume Frequency (weekly)", ['0-2 times', '3-4 times', '5-7 times'])
    reasons = st.selectbox("Reasons for choosing brands", ['Price', 'Quality', 'Availability', 'Brand Reputation'])
    health = st.selectbox("Health Concerns", [
        'Low (Not very concerned)',
        'Medium (Moderately health-conscious)',
        'High (Very health-conscious)'
    ])

# -------- COLUMN 3 --------
with col3:
    zone = st.selectbox("Zone", ['Urban', 'Metro', 'Rural', 'Semi-Urban'])
    brand = st.selectbox("Current Brand", ['Newcomer', 'Established'])
    flavor = st.selectbox("Flavor Preference", ['Traditional', 'Exotic'])
    situation = st.selectbox("Typical Consumption Situations", [
        'Active (eg. Sports, gym)',
        'Casual (eg. At home)',
        'Social (eg. Parties)'
    ])

# -------- COLUMN 4 --------
with col4:
    occupation = st.selectbox("Occupation", ['Working Professional', 'Student', 'Entrepreneur', 'Retired'])
    size = st.selectbox("Preferable Consumption Size", [
        'Small (250 ml)',
        'Medium (500 ml)',
        'Large (1 L)'
    ])
    channel = st.selectbox("Purchase Channel", ['Online', 'Retail Store'])

# Button
if st.button("Calculate Price Range"):

    # Convert age → age_group (same logic as training)
    def get_age_group(age):
        if age <= 25:
            return '18-25'
        elif age <= 35:
            return '26-35'
        elif age <= 45:
            return '36-45'
        elif age <= 55:
            return '46-55'
        elif age <= 70:
            return '56-70'
        else:
            return '70+'


    age_group = get_age_group(age)

    # Create dataframe
    input_df = pd.DataFrame([{
        'age_group': age_group,
        'gender': gender,
        'zone': zone,
        'occupation': occupation,
        'income_levels': income,
        'consume_frequency(weekly)': frequency,
        'current_brand': brand,
        'preferable_consumption_size': size,
        'awareness_of_other_brands': awareness,
        'reasons_for_choosing_brands': reasons,
        'flavor_preference': flavor,
        'purchase_channel': channel,
        'packaging_preference': packaging,
        'health_concerns': health,
        'typical_consumption_situations': situation
    }])

    # ---- Feature Engineering (same as training) ----

    def create_features(df):
        df = df.copy()

        # Income score
        income_map = {
            '<10L': 1,
            '10L - 15L': 2,
            '16L - 25L': 3,
            '26L - 35L': 4,
            '> 35L': 5,
            'Not Reported': 0
        }

        df['income_score'] = df['income_levels'].map(income_map)

        # Frequency score
        freq_map = {
            '0-2 times': 1,
            '3-4 times': 2,
            '5-7 times': 3
        }
        df['frequency_score'] = df['consume_frequency(weekly)'].map(freq_map)

        # Awareness score
        awareness_map = {
            '0 to 1': 1,
            '2 to 4': 2,
            'above 4': 3
        }
        df['awareness_score'] = df['awareness_of_other_brands'].map(awareness_map)

        # Zone score
        zone_map = {
            'Urban': 3,
            'Metro': 4,
            'Rural': 1,
            'Semi-Urban': 2
        }

        df['zone_score'] = df['zone'].map(zone_map)

        # Example derived features (match your notebook logic)
        df['cf_ab_score'] = round(df['frequency_score'] / (df['awareness_score'] + df['frequency_score']), 2)

        df['zas_score'] = df['zone_score'] * df['income_score']

        df['bsi'] = ((df['current_brand'] != 'Established') & (df['reasons_for_choosing_brands'].isin(['Price', 'Quality']))).astype(int)

        return df


    # Prediction
    input_df = create_features(input_df)
    pred = model.predict(input_df)
    final_output = le.inverse_transform(pred)

    st.success(f"Predicted Price Range: {final_output[0]}")
