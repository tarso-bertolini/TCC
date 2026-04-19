import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_microgrid_data(start_date: str = "2025-01-01", days: int = 365) -> pd.DataFrame:
    """
    Generates synthetic historical telemetry data for a microgrid simulation.
    
    Args:
        start_date (str): The start date of the dataset in YYYY-MM-DD format.
        days (int): Number of days to generate data for.
        
    Returns:
        pd.DataFrame: A DataFrame containing timestamp, pv_kw, demand_kw, and tariff_rate.
    """
    np.random.seed(42)
    hours = days * 24
    timestamps = pd.date_range(start=start_date, periods=hours, freq="h")
    
    # 1. PV Generation (kW): Diurnal cycles (bell curve during the day), seasonal variation, and stochastic cloud cover
    hour_of_day = timestamps.hour
    day_of_year = timestamps.dayofyear
    
    # Seasonal multiplier (higher in summer, lower in winter) - Assuming Northern Hemisphere for simplicity
    seasonal_multiplier = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Base PV (only active between 6 AM and 6 PM roughly)
    pv_base = np.clip(np.sin((hour_of_day - 6) * np.pi / 12), 0, 1) * 5.0 # Max 5 kW
    
    # Add noise (cloud cover)
    cloud_noise = np.random.uniform(0.5, 1.0, size=hours)
    pv_kw = pv_base * seasonal_multiplier * cloud_noise
    
    # 2. Demand (kW): Base load + morning/evening peaks + noise
    demand_base = 1.0 # Base human consumption
    # Morning peak around 8 AM, Evening peak around 7 PM
    morning_peak = np.exp(-0.5 * ((hour_of_day - 8) / 1.5)**2) * 2.0
    evening_peak = np.exp(-0.5 * ((hour_of_day - 19) / 2.0)**2) * 3.0
    
    demand_noise = np.random.normal(0, 0.2, size=hours)
    demand_kw = np.clip(demand_base + morning_peak + evening_peak + demand_noise, 0.5, 10.0)
    
    # 3. Tariff Rate ($/kWh): Time-of-Use with occasional spikes
    tariff_rate = np.zeros(hours)
    
    for i, h in enumerate(hour_of_day):
        if 16 <= h <= 21:
            tariff_rate[i] = 0.35 # On-peak
        elif (7 <= h <= 15) or (22 <= h <= 23):
            tariff_rate[i] = 0.20 # Mid-peak
        else:
            tariff_rate[i] = 0.10 # Off-peak
            
    # Add occasional anomalies (price spikes)
    spike_indices = np.random.choice(hours, size=int(hours * 0.01), replace=False)
    tariff_rate[spike_indices] *= np.random.uniform(2.0, 5.0, size=len(spike_indices))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "pv_kw": pv_kw,
        "demand_kw": demand_kw,
        "tariff_rate": tariff_rate
    })
    
    return df

if __name__ == "__main__":
    df = generate_microgrid_data()
    print(df.head(24))
    df.to_csv("microgrid_data.csv", index=False)
