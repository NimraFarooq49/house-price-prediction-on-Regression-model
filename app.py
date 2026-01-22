import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Ice Cream Revenue Prediction",
    page_icon="🍦",
    layout="wide"
)

# ===============================
# SAFE IMPORT: SCIKIT-LEARN
# ===============================
try:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    sklearn_available = True
except Exception as e:
    sklearn_available = False
    st.error("❌ Scikit-learn is not available.")
    st.error("👉 Fix: Add **scikit-learn** to requirements.txt")
    st.stop()

# ===============================
# SAFE IMPORT: PLOTLY (OPTIONAL)
# ===============================
try:
    import plotly.express as px
    import plotly.graph_objects as go
    plotly_available = True
except Exception:
    plotly_available = False
    st.warning("⚠️ Plotly not available. Using basic Streamlit charts.")

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <div style="background-color:#eef6ff;padding:20px;border-radius:10px;text-align:center;">
        <h1>🍦 Ice Cream Revenue Prediction</h1>
        <p>Machine Learning Dashboard (Lab Project)</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Ice_Cream.csv")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    return df.dropna()

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ===============================
# BASIC MODEL (ONLY IF SKLEARN OK)
# ===============================
X = df[["Temperature"]]
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

st.subheader("📊 Model Performance")
st.write("MAE:", mean_absolute_error(y_test, pred))
st.write("RMSE:", mean_squared_error(y_test, pred, squared=False))
st.write("R² Score:", r2_score(y_test, pred))

# ===============================
# VISUALIZATION
# ===============================
st.subheader("📈 Revenue vs Temperature")

if plotly_available:
    fig = px.scatter(df, x="Temperature", y="Revenue", title="Revenue vs Temperature")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(df.set_index("Temperature")["Revenue"])

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<center>Developed by <b>Safoora</b> | Streamlit ML App</center>",
    unsafe_allow_html=True
)
