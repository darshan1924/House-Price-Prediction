import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        padding: 16px;
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #1a3c34;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        with open('model_compatible.pkl', 'rb') as file:
            model = pickle.load(file)
        data = pd.read_csv('Cleaned_data.csv')
        return model, data
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None

model, data = load_resources()

if model is None or data is None:
    st.stop()

st.title("🏠 Bangalore House Price Predictor")
with st.container():
    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox(
            "📍 Select Location",
            options=sorted(data['location'].unique())
        )

        sqft = st.number_input(
            "🏡 Total Square Feet",
            min_value=300.0,
            max_value=10000.0,
            step=10.0,
            format="%.0f"
        )

    with col2:
        bedrooms = st.number_input(
            "🛏️ Number of Bedrooms",
            min_value=1,
            max_value=10,
            step=1,
            format="%d"
        )

        bathrooms = st.number_input(
            "🛁 Number of Bathrooms",
            min_value=1,
            max_value=10,
            step=1,
            format="%d"
        )

        balcony = st.number_input(
            "🌿 Number of Balconies",
            min_value=0,
            max_value=5,
            step=1,
            format="%d"
        )

if st.button("🔍 Predict Price", use_container_width=True):
    with st.spinner("Calculating price..."):
        input_df = pd.DataFrame([{
            'location': location,
            'total_sqft': sqft,
            'bath': bathrooms,
            'balcony': balcony,
            'bedrooms': bedrooms
        }])

        prediction = model.predict(input_df)[0]
        price_in_lakhs = round(prediction, 2)
        price_str = f"₹ {price_in_lakhs:,.2f} Lakhs"

        st.success(f"**Estimated Price:** {price_str}")