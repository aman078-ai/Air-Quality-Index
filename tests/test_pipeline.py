import pytest
import pandas as pd
import numpy as np
from pipeline import preprocess_sensor_data

def test_preprocess_sensor_data_imputation():
    # Create dummy data with some missing values
    data = {
        "timestamp": pd.date_range(start="2026-01-01 00:00:00", periods=5, freq="h"),
        "PM2.5": [10.0, np.nan, 30.0, np.nan, 50.0],
        "Temperature": [np.nan, 20.0, np.nan, 22.0, np.nan],
        "Humidity": [80.0, 81.0, 82.0, 83.0, 84.0]
    }
    df = pd.DataFrame(data)
    
    # Process
    processed_df = preprocess_sensor_data(df)
    
    # Assertions
    # 1. No missing values left in numeric columns
    assert not processed_df["PM2.5"].isnull().any()
    assert not processed_df["Temperature"].isnull().any()
    
    # 2. Linear interpolation checks
    # PM2.5: [10.0, nan, 30.0, nan, 50.0] -> interpolation should yield [10, 20, 30, 40, 50]
    np.testing.assert_array_almost_equal(processed_df["PM2.5"].values, [10.0, 20.0, 30.0, 40.0, 50.0])
    
    # Temperature boundary filling checks: [nan, 20, nan, 22, nan]
    # Interpolation gives [nan, 20, 21, 22, nan]
    # ffill/bfill should give [20, 20, 21, 22, 22]
    np.testing.assert_array_almost_equal(processed_df["Temperature"].values, [20.0, 20.0, 21.0, 22.0, 22.0])

def test_preprocess_sensor_data_temporal_features():
    # Create test data spanning weekday/weekend and specific hours/months
    # Monday is 0, Sunday is 6
    # 2026-01-01 is a Thursday (day_of_week = 3)
    # 2026-01-04 is a Sunday (day_of_week = 6)
    data = {
        "timestamp": [
            "2026-01-01 00:00:00", # Thursday, Jan, Hour 0
            "2026-01-04 12:00:00", # Sunday, Jan, Hour 12
            "2026-07-15 18:00:00"  # Wednesday, Jul, Hour 18
        ],
        "PM2.5": [12.0, 15.0, 18.0]
    }
    df = pd.DataFrame(data)
    
    processed_df = preprocess_sensor_data(df)
    
    # Verify columns exist
    expected_cols = ["hour", "day_of_week", "month", "day", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"]
    for col in expected_cols:
        assert col in processed_df.columns
        
    # Check is_weekend
    assert processed_df.loc[0, "is_weekend"] == 0 # Thursday
    assert processed_df.loc[1, "is_weekend"] == 1 # Sunday
    assert processed_df.loc[2, "is_weekend"] == 0 # Wednesday
    
    # Check hours
    assert processed_df.loc[0, "hour"] == 0
    assert processed_df.loc[1, "hour"] == 12
    assert processed_df.loc[2, "hour"] == 18
    
    # Check cyclical Hour 0: sin(0) = 0, cos(0) = 1
    assert pytest.approx(processed_df.loc[0, "hour_sin"]) == 0.0
    assert pytest.approx(processed_df.loc[0, "hour_cos"]) == 1.0
    
    # Check cyclical Hour 12: sin(pi) = 0, cos(pi) = -1
    assert pytest.approx(processed_df.loc[1, "hour_sin"]) == 0.0
    assert pytest.approx(processed_df.loc[1, "hour_cos"]) == -1.0
