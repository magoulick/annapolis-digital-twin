"""
Note: This script demonstrates the data processing methodology used in the research.
The actual sample_data.csv contains real observations from the full dataset.
Run this script to understand the processing pipeline, not to recreate the sample data.
"""

#!/usr/bin/env python3
"""
Annapolis Digital Twin Data Processing Pipeline
===============================================

This script demonstrates the complete data processing pipeline used to create
the 508,000-record training dataset for the Annapolis Digital Twin.

Author: Paul Magoulick
Email: magoulic@usna.edu
Institution: United States Naval Academy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import requests
import warnings
from scipy import interpolate
from scipy.stats import zscore
import json

warnings.filterwarnings('ignore')

class AnnapolisDataProcessor:
    """
    Main data processing class for the Annapolis Digital Twin
    """
    
    def __init__(self, db_path='annapolis_data.db'):
        self.db_path = db_path
        self.station_id = '8575512'  # NOAA Annapolis
        self.usgs_site = '01647000'  # Severn River
        
        # Data quality parameters
        self.outlier_threshold = 3.5  # Modified Z-score threshold
        self.max_gap_minutes = 30     # Maximum gap for interpolation
        self.water_level_range = (-3.0, 10.0)  # Physically plausible range (ft)
        
    def collect_noaa_data(self, start_date, end_date):
        """
        Collect water level and meteorological data from NOAA
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pandas.DataFrame
            NOAA data with timestamp index
        """
        print(f"Collecting NOAA data from {start_date} to {end_date}")
        
        # NOAA API parameters
        base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        
        # Water level data
        wl_params = {
            'station': self.station_id,
            'product': 'water_level',
            'datum': 'MLLW',
            'time_zone': 'gmt',
            'format': 'json',
            'begin_date': start_date.replace('-', ''),
            'end_date': end_date.replace('-', '')
        }
        
        # Meteorological data
        met_params = {
            'station': self.station_id,
            'product': 'meteorological',
            'time_zone': 'gmt',
            'format': 'json',
            'begin_date': start_date.replace('-', ''),
            'end_date': end_date.replace('-', '')
        }
        
        try:
            # Fetch water level data
            wl_response = requests.get(base_url, params=wl_params, timeout=30)
            wl_data = wl_response.json()
            
            # Fetch meteorological data
            met_response = requests.get(base_url, params=met_params, timeout=30)
            met_data = met_response.json()
            
            # Process water level data
            wl_df = pd.DataFrame(wl_data.get('data', []))
            if not wl_df.empty:
                wl_df['timestamp'] = pd.to_datetime(wl_df['t'])
                wl_df['water_level_ft'] = pd.to_numeric(wl_df['v'], errors='coerce')
                wl_df = wl_df[['timestamp', 'water_level_ft']].set_index('timestamp')
            
            # Process meteorological data
            met_df = pd.DataFrame(met_data.get('data', []))
            if not met_df.empty:
                met_df['timestamp'] = pd.to_datetime(met_df['t'])
                met_df['wind_speed'] = pd.to_numeric(met_df.get('s', 0), errors='coerce')
                met_df['wind_dir'] = pd.to_numeric(met_df.get('d', 0), errors='coerce')
                met_df['pressure'] = pd.to_numeric(met_df.get('p', 0), errors='coerce')
                met_df = met_df[['timestamp', 'wind_speed', 'wind_dir', 'pressure']].set_index('timestamp')
            
            # Merge datasets
            combined = wl_df.join(met_df, how='outer')
            return combined
            
        except Exception as e:
            print(f"Error collecting NOAA data: {e}")
            return pd.DataFrame()
    
    def collect_usgs_data(self, start_date, end_date):
        """
        Collect stream discharge data from USGS
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pandas.DataFrame
            USGS data with timestamp index
        """
        print(f"Collecting USGS data from {start_date} to {end_date}")
        
        url = f"https://waterservices.usgs.gov/nwis/iv/"
        params = {
            'sites': self.usgs_site,
            'parameterCd': '00060,00065',  # Discharge, Gage height
            'startDT': start_date,
            'endDT': end_date,
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            records = []
            time_series = data['value']['timeSeries']
            
            for series in time_series:
                param_cd = series['variable']['variableCode'][0]['value']
                values = series['values'][0]['value']
                
                for value in values:
                    timestamp = pd.to_datetime(value['dateTime'])
                    val = float(value['value'])
                    
                    if param_cd == '00060':  # Discharge
                        records.append({'timestamp': timestamp, 'discharge_cfs': val})
                    elif param_cd == '00065':  # Gage height
                        records.append({'timestamp': timestamp, 'gage_height_ft': val})
            
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.groupby('timestamp').first()
                
            return df
            
        except Exception as e:
            print(f"Error collecting USGS data: {e}")
            return pd.DataFrame()
    
    def quality_control(self, df):
        """
        Apply quality control procedures to the dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw data with timestamp index
            
        Returns:
        --------
        pandas.DataFrame
            Quality-controlled data
        """
        print("Applying quality control procedures...")
        
        df_qc = df.copy()
        
        # Range checks for water level
        if 'water_level_ft' in df_qc.columns:
            mask = (df_qc['water_level_ft'] >= self.water_level_range[0]) & \
                   (df_qc['water_level_ft'] <= self.water_level_range[1])
            df_qc.loc[~mask, 'water_level_ft'] = np.nan
        
        # Rate of change filter (max 1.64 ft in 6 minutes)
        if 'water_level_ft' in df_qc.columns:
            rate_change = df_qc['water_level_ft'].diff().abs()
            mask = rate_change <= 1.64
            df_qc.loc[~mask, 'water_level_ft'] = np.nan
        
        # Statistical outlier detection (modified Z-score)
        for col in df_qc.select_dtypes(include=[np.number]).columns:
            scores = np.abs(zscore(df_qc[col], nan_policy='omit'))
            df_qc.loc[scores > self.outlier_threshold, col] = np.nan
        
        return df_qc
    
    def interpolate_gaps(self, df):
        """
        Interpolate small gaps in the dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Quality-controlled data
            
        Returns:
        --------
        pandas.DataFrame
            Data with interpolated gaps
        """
        print("Interpolating small data gaps...")
        
        df_interp = df.copy()
        
        for col in df_interp.select_dtypes(include=[np.number]).columns:
            # Only interpolate gaps smaller than max_gap_minutes
            df_interp[col] = df_interp[col].interpolate(method='cubic', limit=5)
        
        return df_interp
    
    def engineer_temporal_features(self, df):
        """
        Create temporal features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data with datetime index
            
        Returns:
        --------
        pandas.DataFrame
            Data with temporal features added
        """
        print("Engineering temporal features...")
        
        df_temp = df.copy()
        
        # Cyclical temporal features
        df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp.index.hour / 24)
        df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp.index.hour / 24)
        df_temp['day_sin'] = np.sin(2 * np.pi * df_temp.index.dayofyear / 365.25)
        df_temp['day_cos'] = np.cos(2 * np.pi * df_temp.index.dayofyear / 365.25)
        
        # Rolling statistics for water level
        if 'water_level_ft' in df_temp.columns:
            windows = [10, 20, 60, 120, 240]  # 1, 2, 6, 12, 24 hours
            for w in windows:
                df_temp[f'water_level_mean_{w}'] = df_temp['water_level_ft'].rolling(w).mean()
                df_temp[f'water_level_std_{w}'] = df_temp['water_level_ft'].rolling(w).std()
                df_temp[f'water_level_max_{w}'] = df_temp['water_level_ft'].rolling(w).max()
        
        # Key persistence features (most important for predictions)
        if 'water_level_ft' in df_temp.columns:
            df_temp['max_water_level_24h'] = df_temp['water_level_ft'].rolling(240).max()
            df_temp['max_water_level_12h'] = df_temp['water_level_ft'].rolling(120).max()
            df_temp['max_water_level_48h'] = df_temp['water_level_ft'].rolling(480).max()
        
        return df_temp
    
    def engineer_meteorological_features(self, df):
        """
        Create meteorological features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with meteorological features
        """
        print("Engineering meteorological features...")
        
        df_met = df.copy()
        
        if 'wind_speed' in df_met.columns and 'wind_dir' in df_met.columns:
            # Wind stress components (Large and Pond, 1981)
            rho_air = 1.225  # kg/m^3
            
            # Drag coefficient calculation
            def calc_drag_coefficient(wind_speed):
                if wind_speed < 11:
                    return 1.2e-3
                elif wind_speed < 25:
                    return (0.49 + 0.065 * wind_speed) * 1e-3
                else:
                    return 2.3e-3
            
            df_met['drag_coeff'] = df_met['wind_speed'].apply(calc_drag_coefficient)
            
            # Wind stress components
            wind_rad = np.radians(df_met['wind_dir'])
            df_met['wind_stress_x'] = (rho_air * df_met['drag_coeff'] * 
                                     df_met['wind_speed']**2 * np.cos(wind_rad))
            df_met['wind_stress_y'] = (rho_air * df_met['drag_coeff'] * 
                                     df_met['wind_speed']**2 * np.sin(wind_rad))
        
        # Pressure trends
        if 'pressure' in df_met.columns:
            df_met['pressure_trend'] = df_met['pressure'].diff()
            df_met['pressure_trend_6h'] = df_met['pressure'].diff(60)
        
        return df_met
    
    def identify_extreme_events(self, df, threshold=2.6):
        """
        Identify extreme flood events
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        threshold : float
            Water level threshold for flood events (ft)
            
        Returns:
        --------
        pandas.DataFrame
            Data with extreme event flags
        """
        print(f"Identifying extreme events (threshold: {threshold} ft)")
        
        df_events = df.copy()
        
        if 'water_level_ft' in df_events.columns:
            df_events['flood_stage'] = df_events['water_level_ft'] > threshold
            
            # Identify discrete extreme events
            df_events['extreme_event'] = False
            flood_periods = df_events['flood_stage'].astype(int).diff().fillna(0)
            event_starts = flood_periods == 1
            
            # Mark extreme events (duration > 3 hours or peak > 4.0 ft)
            current_event = False
            event_start_idx = None
            
            for idx, (timestamp, row) in enumerate(df_events.iterrows()):
                if event_starts.iloc[idx]:  # Start of flood event
                    current_event = True
                    event_start_idx = idx
                elif not row['flood_stage'] and current_event:  # End of event
                    current_event = False
                    # Check if event qualifies as extreme
                    event_data = df_events.iloc[event_start_idx:idx+1]
                    duration_hours = len(event_data) * 6 / 60  # 6-minute intervals
                    max_level = event_data['water_level_ft'].max()
                    
                    if duration_hours > 3 or max_level > 4.0:
                        df_events.iloc[event_start_idx:idx+1, 
                                     df_events.columns.get_loc('extreme_event')] = True
        
        return df_events
    
    def create_sample_dataset(self, df, n_samples=4000):
        """
        Create representative sample dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Full processed dataset
        n_samples : int
            Number of samples to extract
            
        Returns:
        --------
        pandas.DataFrame
            Sample dataset
        """
        print(f"Creating sample dataset with {n_samples} records...")
        
        # Stratified sampling to preserve patterns
        sample_data = []
        
        # Get extreme events (ensure they're represented)
        if 'extreme_event' in df.columns:
            extreme_events = df[df['extreme_event'] == True]
            n_extreme = min(len(extreme_events), int(n_samples * 0.1))  # 10% extreme events
            sample_data.append(extreme_events.sample(n=n_extreme, random_state=42))
        else:
            n_extreme = 0
        
        # Regular sampling for remaining data
        n_regular = n_samples - n_extreme
        regular_data = df[df.get('extreme_event', False) == False]
        
        if len(regular_data) > n_regular:
            # Systematic sampling to preserve temporal patterns
            step = len(regular_data) // n_regular
            indices = range(0, len(regular_data), step)[:n_regular]
            sample_data.append(regular_data.iloc[indices])
        else:
            sample_data.append(regular_data)
        
        # Combine and sort by timestamp
        sample_df = pd.concat(sample_data).sort_index()
        
        print(f"Sample dataset created: {len(sample_df)} records")
        if 'extreme_event' in sample_df.columns:
            print(f"Extreme events in sample: {sample_df['extreme_event'].sum()}")
        
        return sample_df

def main():
    """
    Main processing pipeline demonstration
    """
    print("Annapolis Digital Twin Data Processing Pipeline")
    print("=" * 50)
    
    # Initialize processor
    processor = AnnapolisDataProcessor()
    
    # Example processing for sample period
    start_date = "2024-01-01"
    end_date = "2024-01-31"  # One month for demonstration
    
    print(f"Processing sample period: {start_date} to {end_date}")
    
    # Step 1: Collect raw data
    print("\n1. Data Collection")
    noaa_data = processor.collect_noaa_data(start_date, end_date)
    usgs_data = processor.collect_usgs_data(start_date, end_date)
    
    # Combine datasets
    raw_data = noaa_data.join(usgs_data, how='outer')
    print(f"Raw data collected: {len(raw_data)} records")
    
    # Step 2: Quality control
    print("\n2. Quality Control")
    qc_data = processor.quality_control(raw_data)
    
    # Step 3: Interpolate gaps
    print("\n3. Gap Interpolation")
    interp_data = processor.interpolate_gaps(qc_data)
    
    # Step 4: Feature engineering
    print("\n4. Feature Engineering")
    temporal_features = processor.engineer_temporal_features(interp_data)
    met_features = processor.engineer_meteorological_features(temporal_features)
    
    # Step 5: Identify extreme events
    print("\n5. Extreme Event Identification")
    final_data = processor.identify_extreme_events(met_features)
    
    # Step 6: Create sample dataset
    print("\n6. Sample Dataset Creation")
    sample_data = processor.create_sample_dataset(final_data, n_samples=4000)
    
    # Save sample dataset
    output_file = 'sample_data.csv'
    sample_data.to_csv(output_file)
    print(f"\nSample dataset saved to: {output_file}")
    
    # Display summary statistics
    print(f"\nDataset Summary:")
    print(f"Total records: {len(sample_data)}")
    print(f"Date range: {sample_data.index.min()} to {sample_data.index.max()}")
    print(f"Features: {len(sample_data.columns)}")
    
    if 'water_level_ft' in sample_data.columns:
        wl_stats = sample_data['water_level_ft'].describe()
        print(f"\nWater Level Statistics (ft MLLW):")
        print(f"Mean: {wl_stats['mean']:.2f}")
        print(f"Std: {wl_stats['std']:.2f}")
        print(f"Range: {wl_stats['min']:.2f} to {wl_stats['max']:.2f}")
    
    if 'extreme_event' in sample_data.columns:
        n_extreme = sample_data['extreme_event'].sum()
        print(f"\nExtreme Events: {n_extreme} ({n_extreme/len(sample_data)*100:.1f}%)")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
