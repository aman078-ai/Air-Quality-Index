import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_sensor_data(df, is_training=True):
    """
    Cleans raw air quality sensor data, handles missing values,
    and performs feature engineering.
    """
    df = df.copy()
    
    # 1. Parse timestamp and sort
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Identify numeric columns for imputation (pollutants and meteorology)
    numeric_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "Temperature", "Humidity"]
    # Keep only those present in the dataframe
    cols_to_impute = [col for col in numeric_cols if col in df.columns]
    
    # 2. Handle missing sensor data using linear interpolation
    # Time-series data is continuous, so linear interpolation is ideal.
    # We follow up with ffill and bfill for boundary values.
    df[cols_to_impute] = df[cols_to_impute].interpolate(method='linear', limit_direction='both')
    df[cols_to_impute] = df[cols_to_impute].ffill().bfill()
    
    # 3. Create temporal features
    if 'timestamp' in df.columns:
        # Extract basic features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        # Create cyclical encodings for hour (0-23) and month (1-12)
        # to preserve continuous daily and seasonal transitions
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
    return df

def prepare_data(file_path, test_size=0.2, random_state=42, time_based_split=True):
    """
    Loads raw data, runs preprocessing, splits into train/test sets,
    and scales the feature set.
    """
    print(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Preprocess
    df_clean = preprocess_sensor_data(df)
    
    # Drop rows where target variable AQI is missing (if any)
    if 'AQI' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['AQI'])
    else:
        raise ValueError("Target column 'AQI' not found in dataset.")
        
    # Define features
    feature_cols = [
        "PM2.5", "PM10", "NO2", "CO", "SO2", 
        "Temperature", "Humidity",
        "hour_sin", "hour_cos", 
        "month_sin", "month_cos", 
        "day_of_week", "is_weekend"
    ]
    
    # Verify features exist
    available_features = [col for col in feature_cols if col in df_clean.columns]
    
    X = df_clean[available_features]
    y = df_clean['AQI']
    
    # Time-based split or random split
    if time_based_split:
        print("Performing time-based train-test split (preserving sequential order)...")
        split_idx = int(len(df_clean) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        print("Performing random train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    # Standardize input features
    print("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled back to DataFrames for convenience
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=available_features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=available_features)
    
    print(f"Data pipeline processing complete.")
    print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler, available_features

if __name__ == "__main__":
    # Test data pipeline execution
    import os
    data_path = "data/raw_air_quality.csv"
    if not os.path.exists(data_path):
        from generate_data import generate_synthetic_aqi_data
        generate_synthetic_aqi_data()
        
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(data_path)
    print("Features used:", features)
    print("Sample preprocessed training row:")
    print(X_train.head(1))
