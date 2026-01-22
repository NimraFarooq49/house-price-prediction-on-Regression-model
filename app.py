import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

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
        background-color: #f8f9fa;
        border-radius: 10px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1>🏠 House Price Predictor</h1>
        <p>Powered by Decision Tree Machine Learning | Train & Predict in Real-Time</p>
    </div>
""", unsafe_allow_html=True)

# Data Loading Function
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv('House_Price_Prediction_Dataset.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ House_Price_Prediction_Dataset.csv not found. Using sample dummy data for demonstration.")
        data = {
            'Id': range(1, 101),
            'Price': np.random.randint(100000, 1000000, 100),
            'Area': np.random.randint(800, 3500, 100),
            'Bedrooms': np.random.randint(1, 6, 100),
            'Bathrooms': np.random.randint(1, 4, 100),
            'Floors': np.random.randint(1, 3, 100),
            'YearBuilt': np.random.randint(1980, 2024, 100),
            'Location': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 100),
            'Condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 100),
            'Garage': np.random.choice(['Yes', 'No'], 100)
        }
        return pd.DataFrame(data)

df = load_data()

# OneHotEncoder Helper
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.markdown("---")
    
    with st.expander("🔧 Tree Parameters", expanded=True):
        max_depth = st.slider('Max Depth', min_value=1, max_value=30, value=6, 
                              help="Maximum depth of the decision tree")
        min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=2,
                                      help="Minimum samples required to split a node")
        criterion = st.selectbox('Criterion', options=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                help="Function to measure split quality")
    
    with st.expander("📊 Data Split", expanded=True):
        test_size = st.slider('Test Size (%)', 5, 50, 20, 
                              help="Percentage of data used for testing")
        random_state = st.number_input('Random State', value=42, step=1,
                                       help="Seed for reproducibility")
    
    st.markdown("---")
    retrain = st.button('🔄 Retrain Model', type="primary", use_container_width=True)
    show_raw = st.checkbox('📋 Show Raw Data', value=False)
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Info")
    st.info(f"**Rows:** {df.shape[0]:,}  \n**Columns:** {df.shape[1]}")

# Main Content Area
tab1, tab2, tab3 = st.tabs(["📊 Overview & Visualization", "🎯 Model Performance", "🔮 Make Predictions"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Area vs Price Distribution")
        fig_scatter = px.scatter(
            df, x='Area', y='Price', 
            color='Location', 
            size='Bedrooms',
            hover_data=['Bedrooms', 'Bathrooms', 'YearBuilt'],
            height=450,
            title="House Prices by Area and Location"
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Price Distribution")
        fig_hist = px.histogram(
            df, x='Price', 
            nbins=30, 
            title='Price Frequency',
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    if show_raw:
        st.markdown("### 🗂️ Raw Dataset Sample")
        st.dataframe(df.sample(min(200, len(df))).reset_index(drop=True), use_container_width=True)

with tab2:
    if df.shape[0] >= 2:
        with st.spinner('⏳ Training model... Please wait.'):
            try:
                # Prepare data
                X = df.drop(['Id', 'Price'], axis=1)
                y = df['Price']
                categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

                # Build pipeline
                preprocessor = ColumnTransformer([
                    ('cat', make_ohe(), categorical_cols)
                ], remainder='passthrough')

                dt = DecisionTreeRegressor(
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split, 
                    criterion=criterion, 
                    random_state=int(random_state)
                )
                pipeline = Pipeline(steps=[('pre', preprocessor), ('dt', dt)])

                # Train-test split
                actual_test_size = max(0.01, min(test_size/100.0, 1.0 - (1/df.shape[0])))
                if actual_test_size < 0.1 and df.shape[0] > 10:
                    actual_test_size = 0.2

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=actual_test_size, random_state=int(random_state)
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                st.markdown("### 📊 Performance Metrics")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric('Mean Absolute Error', f"${mae:,.0f}")
                col_b.metric('Root Mean Squared Error', f"${rmse:,.0f}")
                col_c.metric('R² Score', f"{r2:.3f}")
                col_d.metric('Test Samples', f"{len(y_test)}")

                # Actual vs Predicted Plot
                st.markdown("### 🎯 Actual vs Predicted Prices")
                fig_ap = go.Figure()
                fig_ap.add_trace(go.Scatter(
                    x=y_test, y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, color='#667eea', opacity=0.6)
                ))
                fig_ap.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red', width=2)
                ))
                fig_ap.update_layout(
                    xaxis_title='Actual Price',
                    yaxis_title='Predicted Price',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=450
                )
                st.plotly_chart(fig_ap, use_container_width=True)

                # Feature Importances
                st.markdown("### 🔍 Feature Importance Analysis")
                try:
                    ohe = pipeline.named_steps['pre'].named_transformers_['cat']
                    if hasattr(ohe, 'get_feature_names_out'):
                        ohe_features = list(ohe.get_feature_names_out(categorical_cols))
                    else:
                        ohe_features = []
                        for i, col in enumerate(categorical_cols):
                            try:
                                categories = ohe.categories_[i]
                            except IndexError:
                                categories = df[col].astype(str).unique()
                            for v in categories:
                                ohe_features.append(f"{col}_{v}")
                    
                    all_features = ohe_features + numeric_cols
                    importances = pipeline.named_steps['dt'].feature_importances_
                    feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(15)
                    
                    fig_fi = px.bar(
                        x=feat_imp.values, 
                        y=feat_imp.index, 
                        orientation='h',
                        labels={'x': 'Importance Score', 'y': 'Feature'},
                        title='Top 15 Most Important Features',
                        color=feat_imp.values,
                        color_continuous_scale='Viridis'
                    )
                    fig_fi.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                except Exception as e:
                    st.warning(f'⚠️ Could not compute feature importances: {str(e)}')
                    
                # Store pipeline in session state for predictions
                st.session_state['trained_pipeline'] = pipeline
                st.session_state['feature_cols'] = X.columns.tolist()
                
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
    else:
        st.error("❌ Insufficient data. Need at least 2 rows to train the model.")

with tab3:
    st.markdown("### 🔮 Predict House Price")
    st.markdown("Enter the house features below to get an instant price prediction.")
    
    with st.form(key='predict_form'):
        col1, col2 = st.columns(2)
        
        def safe_median(col, default):
            try:
                return int(df[col].median())
            except:
                return default
        
        with col1:
            area = st.number_input('🏡 Area (sq ft)', min_value=100, max_value=10000, 
                                  value=safe_median('Area', 1500))
            bedrooms = st.number_input('🛏️ Bedrooms', min_value=0, max_value=10, 
                                      value=safe_median('Bedrooms', 3))
            bathrooms = st.number_input('🚿 Bathrooms', min_value=0, max_value=10, 
                                       value=safe_median('Bathrooms', 2))
            floors = st.number_input('🏢 Floors', min_value=1, max_value=10, 
                                    value=safe_median('Floors', 1))
        
        with col2:
            yearbuilt = st.number_input('📅 Year Built', min_value=1800, max_value=2024, 
                                       value=safe_median('YearBuilt', 2000))
            
            location_options = sorted(df['Location'].unique().tolist()) if 'Location' in df.columns else ['A', 'B', 'C']
            condition_options = sorted(df['Condition'].unique().tolist()) if 'Condition' in df.columns else ['Good', 'Fair']
            garage_options = sorted(df['Garage'].unique().tolist()) if 'Garage' in df.columns else ['Yes', 'No']
            
            location = st.selectbox('📍 Location', options=location_options)
            condition = st.selectbox('⭐ Condition', options=condition_options)
            garage = st.selectbox('🚗 Garage', options=garage_options)
        
        submit = st.form_submit_button('💡 Predict Price', type="primary", use_container_width=True)

        if submit:
            if 'trained_pipeline' in st.session_state:
                try:
                    row = pd.DataFrame([{
                        'Area': area,
                        'Bedrooms': bedrooms,
                        'Bathrooms': bathrooms,
                        'Floors': floors,
                        'YearBuilt': yearbuilt,
                        'Location': location,
                        'Condition': condition,
                        'Garage': garage
                    }])
                    
                    pred = st.session_state['trained_pipeline'].predict(row)[0]
                    
                    st.markdown("---")
                    st.markdown("### 🎉 Prediction Result")
                    st.success(f"## Estimated House Price: **${pred:,.0f}**")
                    
                    # Additional insights
                    avg_price = df['Price'].mean()
                    if pred > avg_price:
                        st.info(f"📈 This price is **${pred - avg_price:,.0f}** above the average market price.")
                    else:
                        st.info(f"📉 This price is **${avg_price - pred:,.0f}** below the average market price.")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
            else:
                st.warning("⚠️ Please train the model first by going to the **Model Performance** tab.")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p style="font-size: 0.9em; color: #666;">
            🚀 Built with Streamlit & Machine Learning | Model updates in real-time
        </p>
        <p style="font-size: 1.1em; font-weight: bold; color: #667eea; margin-top: 10px;">
            Developed by SAFOORA
        </p>
    </div>
""", unsafe_allow_html=True)