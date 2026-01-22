import streamlit as st
import pandas as pd
import numpy as np

# Import sklearn modules with error handling
try:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    sklearn_available = True
except ImportError as e:
    st.error(f"❌ Scikit-learn import error: {str(e)}")
    st.error("Please make sure requirements.txt includes: scikit-learn")
    sklearn_available = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    st.warning("Plotly not available. Using basic charts.")
    plotly_available = False

# Page Configuration with Custom Theme
st.set_page_config(
    page_title='House Price Predictor', 
    page_icon='🏠',
    layout='wide', 
    initial_sidebar_state='expanded'
)

# Custom CSS for Better UI
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 3em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2em;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2em;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background-color: #f
