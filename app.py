import os
import datetime
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as ob

# Set page configuration
st.set_page_config(
    page_title="AeroShield | Air Quality Index Prediction Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(30, 41, 59) 90.1%);
    }
    
    /* Header Card style */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Custom AQI Gauge Card */
    .aqi-card {
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        border: 1px solid;
        transition: transform 0.3s ease;
    }
    .aqi-card:hover {
        transform: translateY(-5px);
    }
    
    .aqi-val {
        font-size: 5rem;
        font-weight: 700;
        margin: 10px 0;
        line-height: 1;
    }
    
    .aqi-lbl {
        font-size: 1.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .aqi-desc {
        font-size: 1rem;
        margin-top: 10px;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        flex: 1;
        text-align: center;
    }
    
    .metric-num {
        font-size: 1.8rem;
        font-weight: 600;
        color: #f8fafc;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid transparent;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #38bdf8;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8;
        border-bottom: 2px solid #38bdf8;
    }
    
    </style>
""", unsafe_allow_html=True)

# Helper function to get AQI category and details
def get_aqi_details(aqi):
    if aqi <= 50:
        return {
            "category": "Good",
            "color": "#10b981", # Emerald green
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "border_color": "#10b981",
            "text_color": "#34d399",
            "desc": "Air quality is satisfactory, and air pollution poses little or no risk."
        }
    elif aqi <= 100:
        return {
            "category": "Moderate",
            "color": "#eab308", # Amber/Yellow
            "bg_color": "rgba(234, 179, 8, 0.15)",
            "border_color": "#eab308",
            "text_color": "#facc15",
            "desc": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
        }
    elif aqi <= 150:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "color": "#f97316", # Orange
            "bg_color": "rgba(249, 115, 22, 0.15)",
            "border_color": "#f97316",
            "text_color": "#ffedd5",
            "desc": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        }
    elif aqi <= 200:
        return {
            "category": "Unhealthy",
            "color": "#ef4444", # Red
            "bg_color": "rgba(239, 68, 68, 0.15)",
            "border_color": "#ef4444",
            "text_color": "#fee2e2",
            "desc": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
        }
    elif aqi <= 300:
        return {
            "category": "Very Unhealthy",
            "color": "#a855f7", # Purple
            "bg_color": "rgba(168, 85, 247, 0.15)",
            "border_color": "#a855f7",
            "text_color": "#f3e8ff",
            "desc": "Health alert: The risk of health effects is increased for everyone."
        }
    else:
        return {
            "category": "Hazardous",
            "color": "#7f1d1d", # Dark Red/Maroon
            "bg_color": "rgba(127, 29, 29, 0.2)",
            "border_color": "#991b1b",
            "text_color": "#fef2f2",
            "desc": "Health warning of emergency conditions: The entire population is more likely to be affected."
        }

# Paths to models
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"

# Load the model and scaler
@st.cache_resource
def load_ml_components():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

# Streamlit App UI
st.markdown("""
    <div class="header-card">
        <div class="header-title">🍃 AeroShield AQI Predictor</div>
        <div class="header-subtitle">Real-time Air Quality Index Forecasting and Sensor Trend Analysis powered by Machine Learning.</div>
    </div>
""", unsafe_allow_html=True)

model, scaler = load_ml_components()

# Auto-train functionality if model doesn't exist
if model is None or scaler is None:
    st.warning("⚠️ Machine Learning components are not yet trained or found in models/.")
    st.info("You can trigger the model training script directly below to generate synthetic data and train the model.")
    if st.button("🚀 Train Model Now", use_container_width=True):
        with st.spinner("Training model (generating data, scaling, and fitting XGBoost/RandomForest)..."):
            try:
                from train import train_model
                train_model()
                st.success("🎉 Model trained successfully! Reloading components...")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to train model: {e}")
    st.stop()

# Tabs definition
tab1, tab2 = st.tabs(["🔮 Real-time Prediction", "📊 Trend Analysis"])

# --- TAB 1: REAL-TIME PREDICTION ---
with tab1:
    st.subheader("Predict AQI from Live Sensor Inputs")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("#### 🧪 Pollutant Concentrations")
        
        # Grid layout for pollutant sliders
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            pm25 = st.slider("PM2.5 (Fine Particulate Matter) - µg/m³", min_value=0.0, max_value=500.0, value=35.0, step=0.1, help="Standard safe limit: 12 µg/m³ (US EPA)")
            pm10 = st.slider("PM10 (Coarse Particulate Matter) - µg/m³", min_value=0.0, max_value=600.0, value=70.0, step=0.1, help="Standard safe limit: 54 µg/m³ (US EPA)")
            no2 = st.slider("NO₂ (Nitrogen Dioxide) - µg/m³", min_value=0.0, max_value=250.0, value=40.0, step=0.1)
            
        with p_col2:
            co = st.slider("CO (Carbon Monoxide) - mg/m³", min_value=0.0, max_value=10.0, value=0.8, step=0.01)
            so2 = st.slider("SO₂ (Sulfur Dioxide) - µg/m³", min_value=0.0, max_value=150.0, value=15.0, step=0.1)
            
        st.write("#### 🌡️ Meteorological Factors")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            temp = st.slider("Temperature - °C", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
        with m_col2:
            humidity = st.slider("Relative Humidity - %", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            
        st.write("#### 🕒 Temporal Factors")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            use_current_time = st.checkbox("Use current system date and time", value=True)
        
        if use_current_time:
            now = datetime.datetime.now()
            hour = now.hour
            month = now.month
            day_of_week = now.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
        else:
            with t_col2:
                selected_time = st.time_input("Select Time", datetime.time(12, 0))
                selected_date = st.date_input("Select Date", datetime.date.today())
                hour = selected_time.hour
                month = selected_date.month
                day_of_week = selected_date.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                
        # Calculate cyclical encodings
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)
        
    with col2:
        st.write("#### 🎯 Predicted AQI")
        
        # Build features array
        # Ensure order matches features exactly:
        # ["PM2.5", "PM10", "NO2", "CO", "SO2", "Temperature", "Humidity", 
        #  "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend"]
        input_data = pd.DataFrame([{
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "CO": co,
            "SO2": so2,
            "Temperature": temp,
            "Humidity": humidity,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }])
        
        # Apply scaling
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction = float(np.clip(prediction, 0, 500))
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            prediction = 0
            
        details = get_aqi_details(prediction)
        
        # HTML card for premium styling
        st.markdown(f"""
            <div class="aqi-card" style="background-color: {details['bg_color']}; border-color: {details['border_color']};">
                <div style="font-size: 1.1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Estimated AQI</div>
                <div class="aqi-val" style="color: {details['color']};">{prediction:.1f}</div>
                <div class="aqi-lbl" style="color: {details['color']};">{details['category']}</div>
                <div style="height: 1px; background-color: {details['border_color']}; opacity: 0.3; margin: 15px 0;"></div>
                <div class="aqi-desc" style="color: {details['text_color']};">{details['desc']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add EPA Breakdowns chart
        st.write("")
        st.write("##### EPA Reference Color Chart")
        ref_df = pd.DataFrame({
            "Range": ["0 - 50", "51 - 100", "101 - 150", "151 - 200", "201 - 300", "301+"],
            "Rating": ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"]
        })
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

# --- TAB 2: TREND ANALYSIS ---
with tab2:
    st.subheader("Analyze Historical Sensor Data Trends")
    
    st.write("""
        Upload a historical CSV file with sensor readings to run batch predictions and visualize trends.
        The CSV file should contain columns: `timestamp`, `PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `Temperature`, `Humidity`.
    """)
    
    # Download sample file option
    sample_data_path = "data/raw_air_quality.csv"
    if os.path.exists(sample_data_path):
        sample_df = pd.read_csv(sample_data_path).head(200) # Give 200 rows
        csv_sample = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Historical CSV Data",
            data=csv_sample,
            file_name="sample_air_quality_sensor_data.csv",
            mime="text/csv",
            help="Download a mock dataset to test the CSV batch prediction and trend charts."
        )
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            df_upload = pd.read_csv(uploaded_file)
            
            # Check for required columns
            required_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "Temperature", "Humidity"]
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns in uploaded CSV: {', '.join(missing_cols)}")
            else:
                with st.spinner("Processing dataset and running predictions..."):
                    # Use pipeline's preprocessor
                    from pipeline import preprocess_sensor_data
                    
                    df_proc = preprocess_sensor_data(df_upload)
                    
                    # Ensure timestamp column exists for plotting
                    if 'timestamp' not in df_proc.columns:
                        df_proc['timestamp'] = pd.date_range(start="2026-01-01", periods=len(df_proc), freq='h')
                        
                    # Extract features matching the model definition
                    feature_order = [
                        "PM2.5", "PM10", "NO2", "CO", "SO2", 
                        "Temperature", "Humidity",
                        "hour_sin", "hour_cos", 
                        "month_sin", "month_cos", 
                        "day_of_week", "is_weekend"
                    ]
                    
                    # Ensure columns order is exactly correct
                    X_features = df_proc[feature_order]
                    
                    # Scaler transform
                    X_scaled = scaler.transform(X_features)
                    
                    # Predict AQI
                    preds = model.predict(X_scaled)
                    df_proc["Predicted_AQI"] = np.clip(preds, 0, 500)
                    
                    # Output stats
                    avg_aqi = df_proc["Predicted_AQI"].mean()
                    max_aqi = df_proc["Predicted_AQI"].max()
                    min_aqi = df_proc["Predicted_AQI"].min()
                    
                    st.success("✅ Batch predictions successfully completed!")
                    
                    # Metrics Grid
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-card">
                                <div class="metric-num" style="color: #38bdf8;">{avg_aqi:.1f}</div>
                                <div class="metric-lbl">Average AQI</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-num" style="color: #ef4444;">{max_aqi:.1f}</div>
                                <div class="metric-lbl">Peak AQI</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-num" style="color: #10b981;">{min_aqi:.1f}</div>
                                <div class="metric-lbl">Minimum AQI</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart 1: AQI trend over time
                    st.write("### 📈 AQI Trend Over Time")
                    fig_aqi = px.line(
                        df_proc, 
                        x="timestamp", 
                        y="Predicted_AQI", 
                        title="Predicted AQI Time Series",
                        color_discrete_sequence=["#6366f1"]
                    )
                    fig_aqi.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#f8fafc",
                        xaxis=dict(showgrid=True, gridcolor="#334155"),
                        yaxis=dict(showgrid=True, gridcolor="#334155")
                    )
                    st.plotly_chart(fig_aqi, use_container_width=True)
                    
                    # Chart 2: Pollutant breakdowns
                    st.write("### 🧪 Pollutant Concentrations Trends")
                    pollutant_choice = st.selectbox(
                        "Select a pollutant to plot alongside Temperature:",
                        ["PM2.5", "PM10", "NO2", "CO", "SO2", "Humidity"]
                    )
                    
                    fig_poll = px.line(
                        df_proc,
                        x="timestamp",
                        y=[pollutant_choice, "Temperature"],
                        title=f"{pollutant_choice} and Temperature Over Time",
                        color_discrete_map={pollutant_choice: "#38bdf8", "Temperature": "#fb7185"}
                    )
                    fig_poll.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#f8fafc",
                        xaxis=dict(showgrid=True, gridcolor="#334155"),
                        yaxis=dict(showgrid=True, gridcolor="#334155")
                    )
                    st.plotly_chart(fig_poll, use_container_width=True)
                    
                    # Chart 3: Correlation Matrix
                    st.write("### 🔗 Feature Correlation Heatmap")
                    corr_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "Temperature", "Humidity", "Predicted_AQI"]
                    corr_matrix = df_proc[corr_cols].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Correlation Coefficient Matrix"
                    )
                    fig_corr.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#f8fafc"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Show raw table
                    st.write("### 📋 Preview Predicted Data")
                    st.dataframe(df_proc[["timestamp"] + required_cols + ["Predicted_AQI"]].head(100), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")
            st.exception(e)
