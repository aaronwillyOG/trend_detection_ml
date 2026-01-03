import streamlit as st
import requests
import json

# Define the API URL (Localhost)
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="BTC Trend Detector", page_icon="📈")

# --- HEADER ---
st.title("⚡ Bitcoin Real-Time Trend Detector")
st.markdown("Enter market indicators below to predict if BTC will go **UP** or **DOWN** in the next 15 mins.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Market Indicators")

def user_input_features():
    # We use default values close to real data so it looks good initially
    ma_10 = st.sidebar.number_input("MA_10 (Short Term Trend)", value=45000.0)
    ma_50 = st.sidebar.number_input("MA_50 (Long Term Trend)", value=44500.0)
    rsi = st.sidebar.slider("RSI (Momentum)", 0.0, 100.0, 55.0)
    volatility = st.sidebar.number_input("Volatility (Std Dev)", value=100.0)
    log_return = st.sidebar.number_input("Log Return", value=0.001, format="%.5f")
    
    data = {
        "ma_10": ma_10,
        "ma_50": ma_50,
        "rsi": rsi,
        "volatility": volatility,
        "log_return": log_return
    }
    return data

input_data = user_input_features()

# --- PREDICTION BUTTON ---
if st.button("🚀 Predict Trend"):
    try:
        # Send request to FastAPI
        response = requests.post(API_URL, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            confidence = result['confidence']
            
            # Display Result
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction")
                if "UP" in prediction:
                    st.success(f"## {prediction}")
                else:
                    st.error(f"## {prediction}")
            
            with col2:
                st.subheader("Confidence")
                st.metric(label="Model Certainty", value=f"{confidence*100:.1f}%")
                st.progress(confidence)
                
            # Show raw data for debugging
            with st.expander("See Raw API Response"):
                st.json(result)
                
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to API. Is 'uvicorn' running?")