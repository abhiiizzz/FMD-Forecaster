from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

import sys

# Additional imports for heatmap functionality
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os
from datetime import datetime

app = Flask(__name__)

# ------------------------------
# Function Definitions
# ------------------------------

# Function to remove outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_shape = df.shape[0]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            final_shape = df.shape[0]
            print(f"Removed {initial_shape - final_shape} outliers from '{col}'")
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return df

# Function to plot time series data
def plot_time_series(data, factor):
    plt.figure(figsize=(12,6))
    plt.plot(data['date'], data[factor], marker='o')
    plt.title(f'Time Series of {factor}')
    plt.xlabel('Date')
    plt.ylabel(factor)
    plt.grid(True)
    plt.close()  # Prevents displaying plots in the Flask app

# Function to perform stationarity check using Augmented Dickey-Fuller test
def check_stationarity(data, factor):
    result = adfuller(data[factor])
    print(f"\nADF Statistic for {factor}: {result[0]:.4f}")
    print(f"p-value for {factor}: {result[1]:.4f}")
    for key, value in result[4].items():
        print(f"Critical Value {key}: {value:.4f}")
    if result[1] < 0.05:
        print(f"Result: The series '{factor}' is stationary.\n")
    else:
        print(f"Result: The series '{factor}' is non-stationary.\n")

# Function to find the optimal SARIMAX parameters using grid search
def get_optimal_sarimax_params(data, factor):
    ts_data = data.groupby('date')[factor].mean().to_frame()
    ts_data = ts_data.asfreq('MS')  # Monthly start frequency
    ts_data = ts_data.fillna(method='ffill')  # Forward fill missing values

    print(f"\nDetermining optimal SARIMAX parameters for {factor}...")
    
    # Define the p, d, q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    
    # Seasonal parameters
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                mod = SARIMAX(ts_data[factor],
                              order=param,
                              seasonal_order=seasonal_param,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
                    best_seasonal_order = seasonal_param
            except:
                continue
    
    print(f"Optimal SARIMAX parameters for {factor}: order={best_order}, seasonal_order={best_seasonal_order}")
    return best_order, best_seasonal_order

# Function to forecast climatic factors using SARIMAX
def forecast_climatic_factor(data, factor, pred_date, order, seasonal_order):
    ts_data = data.groupby('date')[factor].mean().to_frame()
    ts_data = ts_data.asfreq('MS')  # Monthly start frequency
    ts_data = ts_data.fillna(method='ffill')  # Forward fill missing values

    # Fit SARIMAX model with optimal parameters
    try:
        model = SARIMAX(ts_data[factor], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
    except Exception as e:
        print(f"Error fitting SARIMAX model for '{factor}': {e}")
        sys.exit()
    
    # Calculate the number of steps to forecast
    last_date = ts_data.index[-1]
    steps = (pred_date.year - last_date.year) * 12 + (pred_date.month - last_date.month)
    if steps <= 0:
        print("\nPrediction date is within the range of the dataset. Please choose a future date for forecasting.")
        sys.exit()
    
    # Forecast for the specified number of steps
    try:
        forecast = model_fit.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean.iloc[-1]
        return forecast_mean
    except Exception as e:
        print(f"Error forecasting '{factor}': {e}")
        sys.exit()

# Function to perform hyperparameter tuning using RandomizedSearchCV
def tune_random_forest(X, y):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42)
    randomized_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,       # 3-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1    # Utilize all available cores
    )
    randomized_search.fit(X, y)
    print(f"Best Parameters: {randomized_search.best_params_}")
    best_model = randomized_search.best_estimator_
    return best_model

# Function to plot feature importances
def plot_feature_importance(model, feature_names, target_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(f'Feature Importances for {target_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.close()  # Prevents displaying plots in the Flask app

# Function to load and preprocess data
def load_data():
    # Read the Excel file
    excel_file_path = 'data.xlsx'
    
    try:
        df = pd.read_excel(excel_file_path)
        print("Excel file loaded successfully!")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        sys.exit()
    
    # Print all column names to verify
    print("\nColumn Names in the DataFrame:")
    print(df.columns.tolist())
    
    # Create a date column
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1), errors='coerce')
    
    # Check for any parsing errors
    if df['date'].isnull().any():
        print("Error in date conversion. Please check the 'year' and 'month' columns in your Excel file.")
        sys.exit()
    
    # Reset index to make 'date' a column instead of index
    df.reset_index(drop=True, inplace=True)
    
    # Ensure 'district_id' is of string type before applying string methods
    df['district_id'] = df['district_id'].astype(str).str.strip().str.lower()
    
    # Encode district_id
    label_encoder = LabelEncoder()
    df['district_id_code'] = label_encoder.fit_transform(df['district_id'])
    
    # Display available district_id values
    print("\nAvailable district_id values:")
    print(df['district_id'].unique())
    
    # Display the distribution of district_id in the dataset
    print("\nDistribution of district_id in the dataset:")
    print(df['district_id'].value_counts(normalize=True) * 100)
    
    return df, label_encoder

# Function to split data into features and targets
def split_data(df):
    # Remove outliers from specified numerical columns
    numerical_columns = ['Max Temp', 'Min Temp', 'Rainfall', 
                         'number_of_outbreaks', 'number_susceptible', 
                         'number_of_attacks', 'number_of_deaths']
    df = remove_outliers(df, numerical_columns)
    
    # Split the data into features and targets for disease outbreak prediction
    df_features = df[['district_id_code', 'month', 'year', 'Max Temp', 'Min Temp', 'Rainfall', 'date']]
    df_targets = df[['number_of_outbreaks', 'number_susceptible', 'number_of_attacks', 'number_of_deaths']]
    
    # Combine features and targets for time-based splitting
    df_combined = pd.concat([df_features, df_targets], axis=1)
    
    # Ensure the data is sorted by date for time-based splitting
    df_combined.sort_values('date', inplace=True)
    df_combined.reset_index(drop=True, inplace=True)
    
    # Define the split index (80% for training, 20% for testing)
    split_index = int(0.8 * len(df_combined))
    
    # Split the combined DataFrame
    train = df_combined.iloc[:split_index]
    test = df_combined.iloc[split_index:]
    
    # Separate features and targets
    X_train = train[['district_id_code', 'month', 'year', 'Max Temp', 'Min Temp', 'Rainfall']]
    y_train = train[['number_of_outbreaks', 'number_susceptible', 'number_of_attacks', 'number_of_deaths']]
    
    X_test = test[['district_id_code', 'month', 'year', 'Max Temp', 'Min Temp', 'Rainfall']]
    y_test = test[['number_of_outbreaks', 'number_susceptible', 'number_of_attacks', 'number_of_deaths']]
    
    # Handle Missing Values
    if X_train.isnull().any().any() or y_train.isnull().any().any() or X_test.isnull().any().any() or y_test.isnull().any().any():
        print("\nData contains missing values. Filling missing values with mean.")
        X_train.fillna(X_train.mean(), inplace=True)
        y_train.fillna(y_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)
        y_test.fillna(y_test.mean(), inplace=True)
    
    print("\nDataset split:")
    print(f" - Training set size: {X_train.shape[0]} samples ({(X_train.shape[0]/df_combined.shape[0])*100:.2f}%)")
    print(f" - Testing set size: {X_test.shape[0]} samples ({(X_test.shape[0]/df_combined.shape[0])*100:.2f}%)")
    
    return X_train, y_train, X_test, y_test, df_combined

# Function to train and tune RandomForestRegressor models for each target
def train_models(X_train, y_train):
    disease_models = {}
    for target in y_train.columns:
        print(f"\nTraining model for '{target}'...")
        best_model = tune_random_forest(X_train, y_train[target])
        disease_models[target] = best_model
        # Plot Feature Importance
        plot_feature_importance(best_model, X_train.columns, target)
    return disease_models

# Function to evaluate trained models
def evaluate_models(disease_models, X_train, y_train, X_test, y_test):
    print("\nModel Evaluation:")
    for target in y_train.columns:
        # R² on Training Set
        y_train_pred = disease_models[target].predict(X_train)
        r2_train = r2_score(y_train[target], y_train_pred)
        
        # R² on Testing Set
        y_test_pred = disease_models[target].predict(X_test)
        r2_test = r2_score(y_test[target], y_test_pred)
        
        # Calculate additional metrics
        mae = mean_absolute_error(y_test[target], y_test_pred)
        mse = mean_squared_error(y_test[target], y_test_pred)
        
        print(f"\n{target}:")
        print(f" - R² Score on Training Set: {r2_train * 100:.2f}%")
        print(f" - R² Score on Testing Set: {r2_test * 100:.2f}%")
        print(f" - Mean Absolute Error on Testing Set: {mae:.2f}")
        print(f" - Mean Squared Error on Testing Set: {mse:.2f}")

# Function to plot correlation heatmap
def plot_correlation_heatmap(X_train):
    plt.figure(figsize=(8,6))
    sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.close()  # Prevents displaying plots in the Flask app

# ------------------------------
# Initial Data Loading and Model Training
# ------------------------------
df, label_encoder = load_data()
X_train, y_train, X_test, y_test, df_combined = split_data(df)
disease_models = train_models(X_train, y_train)
evaluate_models(disease_models, X_train, y_train, X_test, y_test)
plot_correlation_heatmap(X_train)

# ------------------------------
# Route Definitions
# ------------------------------

# Home Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        district_id_input = request.form['district_id'].strip().lower()
        month_input = int(request.form['month'])
        year_input = int(request.form['year'])
        
        # Redirect to prediction page
        return redirect(url_for('predict', district_id=district_id_input, month=month_input, year=year_input))
    
    # Get list of available districts
    districts = df['district_id'].unique()
    return render_template('index.html', districts=districts)

# Prediction Route for a Single District
@app.route('/predict')
def predict():
    district_id_input = request.args.get('district_id')
    month_input = int(request.args.get('month'))
    year_input = int(request.args.get('year'))
    
    # Check if district_id is valid
    if district_id_input not in df['district_id'].unique():
        return "Invalid district_id. Please enter a valid district_id from the data."
    else:
        # Encode the district_id
        district_id_code_input = label_encoder.transform([district_id_input])[0]
    
    # Filter data for the selected district
    district_data = df[df['district_id'] == district_id_input]
    
    # Ensure that the district has enough data
    # if district_data.shape[0] < 24:  # Increased to 24 for better seasonality capture
    #     return "Not enough data for the selected district to perform time series forecasting. At least 24 data points are required."
    
    # Plot time series and perform stationarity checks
    for factor in ['Max Temp', 'Min Temp', 'Rainfall']:
        plot_time_series(district_data, factor)
        check_stationarity(district_data, factor)
    
    # Get optimal SARIMAX parameters for each climatic factor
    sarimax_params = {}
    for factor in ['Max Temp', 'Min Temp', 'Rainfall']:
        order, seasonal_order = get_optimal_sarimax_params(district_data, factor)
        sarimax_params[factor] = (order, seasonal_order)
    
    # Prepare the prediction date
    try:
        pred_date = pd.Timestamp(year_input, month_input, 1)
    except Exception as e:
        return f"Error creating prediction date: {e}"
    
    # Check if pred_date is beyond the last date in the dataset
    if pred_date <= df_combined['date'].max():
        return "Prediction date is within the range of the dataset. Please choose a future date for forecasting."
    
    # Predict climatic factors with optimal parameters
    predicted_climatic_factors = {}
    for factor in ['Max Temp', 'Min Temp', 'Rainfall']:
        order, seasonal_order = sarimax_params[factor]
        predicted_value = forecast_climatic_factor(district_data, factor, pred_date, order, seasonal_order)
        if factor == 'Rainfall':
            # Ensure rainfall is non-negative
            if predicted_value < 0:
                predicted_value = 0

        predicted_climatic_factors[factor] = predicted_value
    
    print("\nPredicted Climatic Factors:")
    for key, value in predicted_climatic_factors.items():
        print(f"{key}: {value:.2f}")
    
    # Prepare input features for disease prediction
    input_features_disease = pd.DataFrame({
        'district_id_code': [district_id_code_input],
        'month': [month_input],
        'year': [year_input],
        'Max Temp': [predicted_climatic_factors['Max Temp']],
        'Min Temp': [predicted_climatic_factors['Min Temp']],
        'Rainfall': [predicted_climatic_factors['Rainfall']]
    })
    
    # Predict disease data
    predicted_disease_data = {}
    for target in y_train.columns:
        predicted_value = disease_models[target].predict(input_features_disease)[0]
        predicted_disease_data[target] = int(round(predicted_value))  # Round to integer
    
    print("\nPredicted Disease Outbreak Data:")
    for key, value in predicted_disease_data.items():
        print(f"{key}: {value}")
    
    return render_template('prediction.html',
                           climatic_factors=predicted_climatic_factors,
                           disease_data=predicted_disease_data)

@app.route('/heatmap', methods=['GET', 'POST'])
def heatmap():
    if request.method == 'POST':
        try:
            # Retrieve month and year from the form
            month_input = int(request.form['month'])
            year_input = int(request.form['year'])
        except (ValueError, KeyError):
            return "Invalid input for month or year. Please enter numeric values.", 400

        # Validate month and year
        if not (1 <= month_input <= 12):
            return "Invalid month. Please enter a value between 1 and 12.", 400
        if year_input < 1900 or year_input > datetime.now().year + 10:
            return "Invalid year. Please enter a realistic year.", 400

        # Prepare the prediction date
        try:
            pred_date = pd.Timestamp(year_input, month_input, 1)
        except Exception as e:
            return f"Error creating prediction date: {e}", 400

        # Predict climatic factors and outbreaks for all districts
        districts = df['district_id'].unique()
        predicted_outbreaks = []

        for district in districts:
            # Encode the district_id
            try:
                district_code = label_encoder.transform([district])[0]
            except ValueError:
                # If district_id not found in label_encoder
                print(f"District '{district}' not found in label encoder. Skipping.")
                predicted_outbreaks.append({'district_id': district, 'number_of_outbreaks': 0})
                continue

            # Filter data for the current district
            district_data = df[df['district_id'] == district]

            # Ensure that the district has enough data
            if district_data.shape[0] < 24:
                print(f"Not enough data for district '{district}'. Skipping prediction.")
                predicted_outbreaks.append({'district_id': district, 'number_of_outbreaks': 0})
                continue

            # Get optimal SARIMAX parameters for each climatic factor
            sarimax_params = {}
            for factor in ['Max Temp', 'Min Temp', 'Rainfall']:
                order, seasonal_order = get_optimal_sarimax_params(district_data, factor)
                sarimax_params[factor] = (order, seasonal_order)

            # Predict climatic factors
            predicted_climatic_factors = {}
            for factor in ['Max Temp', 'Min Temp', 'Rainfall']:
                order, seasonal_order = sarimax_params[factor]
                predicted_value = forecast_climatic_factor(
                    district_data, factor, pred_date, order, seasonal_order
                )
                if factor == 'Rainfall' and predicted_value < 0:
                    predicted_value = 0  # Ensure rainfall is non-negative
                predicted_climatic_factors[factor] = predicted_value

            # Prepare input features for disease prediction
            input_features_disease = pd.DataFrame({
                'district_id_code': [district_code],
                'month': [month_input],
                'year': [year_input],
                'Max Temp': [predicted_climatic_factors['Max Temp']],
                'Min Temp': [predicted_climatic_factors['Min Temp']],
                'Rainfall': [predicted_climatic_factors['Rainfall']]
            })

            # Predict disease outbreaks
            predicted_disease_data = {}
            for target in y_train.columns:
                predicted_value = disease_models[target].predict(input_features_disease)[0]
                predicted_disease_data[target] = int(round(predicted_value))  # Round to integer

            # Assuming 'number_of_outbreaks' is the primary metric for heatmap
            predicted_outbreaks.append({
                'district_id': district,
                'number_of_outbreaks': predicted_disease_data.get('number_of_outbreaks', 0)
            })

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predicted_outbreaks)

        # Load West Bengal shapefile
        shapefile_path = os.path.join(app.root_path, 'shapefiles', 'District_shape_West_Bengal.shp')

        # Check if shapefile exists
        if not os.path.exists(shapefile_path):
            return "Shapefile not found. Please ensure the path is correct.", 500

        try:
            west_bengal = gpd.read_file(shapefile_path)
        except Exception as e:
            return f"Error reading shapefile: {e}", 500

        # Ensure the shapefile has a 'district_id' column to merge
        shapefile_district_id_column = 'district_id'  # Replace with actual column name if different

        if shapefile_district_id_column not in west_bengal.columns:
            return f"Shapefile does not contain the '{shapefile_district_id_column}' column.", 500

        # Merge predictions with shapefile data
        west_bengal = west_bengal.merge(predictions_df, on='district_id', how='left')
        west_bengal['number_of_outbreaks'] = west_bengal['number_of_outbreaks'].fillna(0)

        # Compute centroids for districts to get latitude and longitude
        west_bengal['centroid'] = west_bengal.geometry.centroid
        west_bengal['lat'] = west_bengal.centroid.y
        west_bengal['lon'] = west_bengal.centroid.x

        # Prepare data_points for HeatMap (latitude, longitude, intensity)
        # Normalize intensity for better visualization
        max_outbreaks = west_bengal['number_of_outbreaks'].max()
        if max_outbreaks == 0:
            max_outbreaks = 1  # To avoid division by zero
        west_bengal['intensity'] = west_bengal['number_of_outbreaks'] / max_outbreaks

        data_points = west_bengal[['lat', 'lon', 'intensity']].values.tolist()

        # Extract bounds for the map
        bounds = west_bengal.total_bounds  # [minx, miny, maxx, maxy]

        # Create a folium map centered around West Bengal
        m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=7)

        # Add West Bengal shapefile to the map
        folium.GeoJson(
            west_bengal,
            name='Districts',
            style_function=lambda feature: {
                'fillColor': '#ffffff',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['district_id', 'number_of_outbreaks'],
                aliases=['District:', 'Number of Outbreaks:'],
                localize=True
            )
        ).add_to(m)

        # Add heatmap layer
        HeatMap(data_points, radius=15, max_zoom=13).add_to(m)

        # Add a legend using branca
        colormap = cm.LinearColormap(['green', 'yellow', 'red'],
                                     vmin=west_bengal['number_of_outbreaks'].min(),
                                     vmax=west_bengal['number_of_outbreaks'].max())
        colormap.caption = 'Number of Predicted Outbreaks'
        colormap.add_to(m)

        # Convert Folium map to HTML representation
        heatmap_html = m._repr_html_()

        return render_template('heatmap_display.html',
                               heatmap=heatmap_html,
                               month=month_input,
                               year=year_input)
    else:
        # Handle GET request: Render the form for user input
        current_year = datetime.now().year
        return render_template('heatmap_form.html', current_year=current_year)
# ------------------------------
# Run the Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
