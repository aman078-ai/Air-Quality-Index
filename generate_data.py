import os
import numpy as np
import pandas as pd

def generate_synthetic_aqi_data():
    print("Generating synthetic air quality dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate hourly timestamps for 2 years (2024 and 2025)
    timestamps = pd.date_range(start="2024-01-01 00:00:00", end="2025-12-31 23:00:00", freq="h")
    n_records = len(timestamps)
    
    # Initialize dictionary for data
    data = {
        "timestamp": timestamps,
    }
    
    # Create base patterns
    # Month (1-12), Hour (0-23)
    months = timestamps.month
    hours = timestamps.hour
    day_of_week = timestamps.dayofweek
    
    # 1. Temperature: seasonal (summer peak around July) + daily cycle (peak at 15:00)
    # Base temp: 22°C + 10°C seasonal wave + 5°C daily wave + noise
    temp_seasonal = 10 * np.sin(2 * np.pi * (months - 4) / 12)
    temp_daily = 5 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp_noise = np.random.normal(0, 2, n_records)
    data["Temperature"] = np.round(20 + temp_seasonal + temp_daily + temp_noise, 1)
    
    # 2. Humidity: inverse relation to temperature
    # Base humidity: 60% - seasonal wave - daily wave + noise
    humidity_seasonal = 15 * np.sin(2 * np.pi * (months - 7) / 12)  # Higher in monsoon/late summer
    humidity_daily = -12 * np.sin(2 * np.pi * (hours - 9) / 24)
    humidity_noise = np.random.normal(0, 5, n_records)
    data["Humidity"] = np.clip(np.round(60 + humidity_seasonal + humidity_daily + humidity_noise, 1), 10, 100)
    
    # Pollutant base trends: high in winter (due to stagnation/heating) and rush hours (traffic)
    # Winter peak factor: winter months (Nov, Dec, Jan, Feb) have high scale
    winter_factor = 1.8 - 0.8 * np.sin(2 * np.pi * (months - 4) / 12)  # Max in Dec/Jan
    
    # Daily rush hour peaks (8 AM and 7 PM)
    rush_hour_factor = 1.0 + 0.4 * np.exp(-((hours - 8) ** 2) / 4) + 0.5 * np.exp(-((hours - 19) ** 2) / 6)
    
    # Weekend reduction factor for traffic pollutants
    weekend_factor = np.where(day_of_week >= 5, 0.8, 1.0)
    
    # 3. PM2.5 (ug/m3): strong seasonal + moderate daily + noise
    pm25_base = 35 * winter_factor * (0.8 + 0.2 * rush_hour_factor)
    pm25_noise = np.random.gamma(shape=3, scale=10, size=n_records) - 30
    data["PM2.5"] = np.clip(np.round(pm25_base + pm25_noise, 1), 1.0, 550.0)
    
    # 4. PM10 (ug/m3): correlates with PM2.5, higher baseline
    pm10_base = data["PM2.5"] * 1.6 + 15
    pm10_noise = np.random.gamma(shape=2, scale=12, size=n_records) - 24
    data["PM10"] = np.clip(np.round(pm10_base + pm10_noise, 1), 2.0, 750.0)
    
    # 5. NO2 (ug/m3): highly traffic dependent (rush hours + weekdays)
    no2_base = 25 * winter_factor * rush_hour_factor * weekend_factor
    no2_noise = np.random.normal(0, 8, n_records)
    data["NO2"] = np.clip(np.round(no2_base + no2_noise, 1), 0.5, 250.0)
    
    # 6. CO (mg/m3): traffic dependent, smaller values
    co_base = 0.8 * winter_factor * rush_hour_factor * weekend_factor
    co_noise = np.random.normal(0, 0.2, n_records)
    data["CO"] = np.clip(np.round(co_base + co_noise, 2), 0.05, 10.0)
    
    # 7. SO2 (ug/m3): industrial pollutant, less daily traffic variance, some winter increase
    so2_base = 12 * winter_factor
    so2_noise = np.random.normal(0, 4, n_records)
    data["SO2"] = np.clip(np.round(so2_base + so2_noise, 1), 0.1, 150.0)
    
    df = pd.DataFrame(data)
    
    # Calculate AQI based on EPA piece-wise linear formulas for pollutants
    # Standard breakpoints for US EPA AQI
    def calc_sub_index(val, breakpoints, aqi_points):
        if val <= breakpoints[0]:
            return aqi_points[0]
        for i in range(len(breakpoints) - 1):
            if breakpoints[i] < val <= breakpoints[i+1]:
                # Interpolate
                c_lo = breakpoints[i]
                c_hi = breakpoints[i+1]
                i_lo = aqi_points[i]
                i_hi = aqi_points[i+1]
                return i_lo + (val - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
        # Beyond the maximum breakpoint
        return aqi_points[-1]
    
    # PM2.5 breakpoints (ug/m3) and AQI points
    pm25_bp = [0.0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]
    aqi_bp = [0, 50, 100, 150, 200, 300, 400, 500]
    
    # PM10 breakpoints (ug/m3)
    pm10_bp = [0.0, 54.0, 154.0, 254.0, 354.0, 424.0, 504.0, 604.0]
    
    # NO2 breakpoints (ug/m3) - roughly converting ppb to ug/m3 (1 ppb NO2 ~ 1.88 ug/m3)
    # ppb bp: 0, 53, 100, 360, 649, 1249
    no2_bp = [0.0, 100.0, 188.0, 676.0, 1220.0, 2348.0, 3100.0, 3850.0]
    
    # CO breakpoints (mg/m3) - roughly 1 ppm ~ 1.15 mg/m3
    # ppm bp: 0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4
    co_bp = [0.0, 5.0, 10.8, 14.2, 17.7, 34.9, 46.4, 57.9]
    
    # SO2 breakpoints (ug/m3) - 1 ppb ~ 2.62 ug/m3
    # ppb bp: 0, 35, 75, 185, 304, 604
    so2_bp = [0.0, 91.7, 196.5, 484.7, 796.5, 1582.5, 2100.0, 2620.0]
    
    print("Calculating true AQI values based on EPA standard breakpoints...")
    aqi_list = []
    for _, row in df.iterrows():
        ipm25 = calc_sub_index(row["PM2.5"], pm25_bp, aqi_bp)
        ipm10 = calc_sub_index(row["PM10"], pm10_bp, aqi_bp)
        ino2 = calc_sub_index(row["NO2"], no2_bp, aqi_bp)
        ico = calc_sub_index(row["CO"], co_bp, aqi_bp)
        iso2 = calc_sub_index(row["SO2"], so2_bp, aqi_bp)
        
        # AQI is the max of individual pollutant sub-indices
        aqi_list.append(max(ipm25, ipm10, ino2, ico, iso2))
        
    df["AQI"] = np.round(aqi_list, 1)
    
    # Inject missing values (NaN) to represent sensor dropouts
    # We inject ~5% missingness randomly in pollutant columns and meteorology
    cols_to_corrupt = ["PM2.5", "PM10", "NO2", "CO", "SO2", "Temperature", "Humidity"]
    for col in cols_to_corrupt:
        mask = np.random.random(n_records) < 0.05
        df.loc[mask, col] = np.nan
        
    # Create target directories if they don't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the synthetic data
    filepath = "data/raw_air_quality.csv"
    df.to_csv(filepath, index=False)
    print(f"Synthetic data generation complete. Saved to '{filepath}'")
    print(f"Generated {len(df)} records. Missing data details:\n{df.isnull().sum()}")

if __name__ == "__main__":
    generate_synthetic_aqi_data()
