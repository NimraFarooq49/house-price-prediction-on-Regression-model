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
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title='House Price Predictor', layout='wide', initial_sidebar_state='collapsed', theme="dark")


@st.cache_data(show_spinner=False)
def load_data():
	try:
		return pd.read_csv('House_Price_Prediction_Dataset.csv')
	except FileNotFoundError:
		st.warning("House_Price_Prediction_Dataset.csv not found. Using dummy data.")
		data = {
			'Id': range(1, 11),
			'Price': np.random.randint(100000, 1000000, 10),
			'Area': np.random.randint(800, 3000, 10),
			'Bedrooms': np.random.randint(1, 6, 10),
			'Bathrooms': np.random.randint(1, 4, 10),
			'Floors': np.random.randint(1, 3, 10),
			'YearBuilt': np.random.randint(1980, 2020, 10),
			'Location': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South'],
			'Condition': ['Good', 'Excellent', 'Fair', 'Good', 'Excellent', 'Fair', 'Good', 'Excellent', 'Fair', 'Good'],
			'Garage': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
		}
		return pd.DataFrame(data)


df = load_data()


def make_ohe():
	try:
		return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
	except TypeError:
		return OneHotEncoder(handle_unknown='ignore', sparse=False)

with st.sidebar:
	st.header('⚙️ Model Controls')
	max_depth = st.slider('Max Depth', min_value=1, max_value=30, value=6)
	min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=2)
	criterion = st.selectbox('Criterion', options=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
	test_size = st.slider('Test Size (%)', 5, 50, 20)
	random_state = st.number_input('Random State', value=42, step=1)
	retrain = st.button('Retrain Model')
	show_raw = st.checkbox('Show Raw Data', value=False)


st.title('🏠 House Price Predictor — Decision Tree Regressor')
st.markdown('A modern interactive frontend that trains a Decision Tree on your dataset **live** and displays metrics and visualizations.')

col1, col2 = st.columns([2, 1])

with col1:
	st.subheader('📊 Dataset Preview')
	if show_raw:
		st.dataframe(df.sample(min(200, len(df))).reset_index(drop=True))
	st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

	st.subheader('Area vs Price')
	fig_scatter = px.scatter(df, x='Area', y='Price', color='Location', hover_data=['Bedrooms','Bathrooms'], height=400)
	st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
	st.subheader('Target Distribution')
	fig_hist = px.histogram(df, x='Price', nbins=40, title='Price Distribution')
	st.plotly_chart(fig_hist, use_container_width=True)


st.markdown('---')

if df.shape[0] >= 2:
	with st.spinner('⏳ Training model...'):
		X = df.drop(['Id','Price'], axis=1)
		y = df['Price']
		categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
		numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

		preprocessor = ColumnTransformer([
			('cat', make_ohe(), categorical_cols)
		], remainder='passthrough')

		dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=int(random_state))
		pipeline = Pipeline(steps=[('pre', preprocessor), ('dt', dt)])

		actual_test_size = max(0.01, min(test_size/100.0, 1.0 - (1/df.shape[0])))
		if actual_test_size < 0.1 and df.shape[0] > 10:
			actual_test_size = 0.2

		try:
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=actual_test_size, random_state=int(random_state))
			pipeline.fit(X_train, y_train)
			y_pred = pipeline.predict(X_test)

			mae = mean_absolute_error(y_test, y_pred)
			mse = mean_squared_error(y_test, y_pred)
			r2 = r2_score(y_test, y_pred)
			
			st.subheader('🚀 Model Performance')
			col_a, col_b, col_c = st.columns(3)
			col_a.metric('MAE', f"**${mae:,.0f}**")
			col_b.metric('MSE', f"**${mse:,.0f}**")
			col_c.metric('R²', f"**{r2:.3f}**")

			st.markdown('### Actual vs Predicted')
			fig_ap = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual Price','y':'Predicted Price'}, title='Actual vs Predicted')
			fig_ap.add_shape(type='line', x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(dash='dash', color='red'))
			st.plotly_chart(fig_ap, use_container_width=True)

			st.markdown('### Feature Importances (Top 20)')
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
				feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(20)
				fig_fi = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h', labels={'x':'Importance','y':'Feature'}, title='Top 20 Feature Importances')
				st.plotly_chart(fig_fi, use_container_width=True)
			except Exception as e:
				st.info('Feature importances not available or error during processing: ' + str(e))
				
		except Exception as e:
			st.error(f"An error occurred during model training/prediction. Please check data and controls. Error: {e}")

else:
	st.error("Insufficient data to train the model. Need at least 2 rows.")


st.markdown('---')
st.subheader('🔍 Interactive Predictions')
with st.form(key='predict_form'):
	st.write('Provide feature values (a few common ones are shown).')
	
	def safe_median(col, default):
		try:
			return int(df[col].median())
		except:
			return default

	area = st.number_input('Area', value=safe_median('Area', 1500))
	bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=safe_median('Bedrooms', 3))
	bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, value=safe_median('Bathrooms', 2))
	floors = st.number_input('Floors', min_value=1, max_value=10, value=safe_median('Floors', 1))
	yearbuilt = st.number_input('Year Built', min_value=1800, max_value=2100, value=safe_median('YearBuilt', 2000))
	
	location_options = sorted(df['Location'].unique().tolist()) if 'Location' in df.columns else ['A', 'B', 'C']
	condition_options = sorted(df['Condition'].unique().tolist()) if 'Condition' in df.columns else ['Good', 'Fair']
	garage_options = sorted(df['Garage'].unique().tolist()) if 'Garage' in df.columns else ['Yes', 'No']
	
	location = st.selectbox('Location', options=location_options)
	condition = st.selectbox('Condition', options=condition_options)
	garage = st.selectbox('Garage', options=garage_options)
	
	submit = st.form_submit_button('Predict Price')

	if submit:
		if 'pipeline' in locals():
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
				if 'Id' in X.columns:
					row['Id'] = 0

				pred = pipeline.predict(row)[0]
				st.success(f'Predicted Price: **${pred:,.0f}**')
			except Exception as e:
				st.error(f"Error during prediction: {e}")
		else:
			st.error("Model has not been trained yet. Please ensure your dataset is available and loaded correctly.")


st.markdown('<small>Model trains live on the dataset and updates when controls change.</small>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; margin-top:20px;"><small>Developed by **SAFOORA**</small></div>', unsafe_allow_html=True)
