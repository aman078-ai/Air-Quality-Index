import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from pipeline import prepare_data

# Define model paths
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

def train_model():
    # Make sure target directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load and prepare data
    raw_data_path = "data/raw_air_quality.csv"
    if not os.path.exists(raw_data_path):
        print(f"Raw data not found at {raw_data_path}. Running generator...")
        from generate_data import generate_synthetic_aqi_data
        generate_synthetic_aqi_data()
        
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(raw_data_path)
    
    # Save the scaler immediately
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    
    # 2. Initialize and train the regressor
    # We will try to use XGBoost, and fallback to RandomForestRegressor if not available.
    try:
        from xgboost import XGBRegressor
        print("Using XGBoost Regressor for training...")
        model = XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    except ImportError:
        print("XGBoost not installed. Falling back to RandomForestRegressor...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        
    print("Training model on scaled features...")
    model.fit(X_train, y_train)
    
    # 3. Perform 5-Fold Cross-Validation on training set
    print("Performing 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train, y_train, 
        cv=kf, 
        scoring=('neg_root_mean_squared_error', 'r2'), 
        return_train_score=False,
        n_jobs=-1
    )
    cv_rmse = -cv_results['test_neg_root_mean_squared_error']
    cv_r2 = cv_results['test_r2']
    
    # 4. Evaluate the model on test set
    print("Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n" + "="*40)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*40)
    print(f"CV RMSE Mean: {cv_rmse.mean():.4f} (std: {cv_rmse.std():.4f})")
    print(f"CV R² Mean:   {cv_r2.mean():.4f} (std: {cv_r2.std():.4f})")
    print("="*40)
    
    print("\n" + "="*40)
    print("MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Train R²:   {train_r2:.4f}")
    print(f"Test R²:    {test_r2:.4f}")
    print("="*40)
    
    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model successfully saved to {MODEL_PATH}")
    
    # 4. Display feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\nFeature Importances:")
        print(feature_imp_df.to_string(index=False))
        
        # Save feature importances to a text file for documentation/reference
        feature_imp_df.to_csv(os.path.join(MODELS_DIR, "feature_importances.csv"), index=False)
        
    return model

if __name__ == "__main__":
    train_model()
