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
