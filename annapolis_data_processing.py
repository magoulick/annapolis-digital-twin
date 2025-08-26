# Annapolis Digital Twin - Data Processing and Feature Engineering
# This script processes the collected data and creates features for machine learning

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class AnnapolisDataProcessor:
    def __init__(self, db_path='annapolis_digital_twin.db'):
        self.db_path = db_path
        self.scaler = StandardScaler()
        
        # Annapolis-specific parameters
        self.tidal_constituents = {
            'M2': 12.42,  # Principal lunar semidiurnal
            'S2': 12.0,   # Principal solar semidiurnal
            'N2': 12.66,  # Lunar elliptic semidiurnal
            'K1': 23.93,  # Lunar diurnal
            'O1': 25.82,  # Lunar diurnal
            'P1': 24.07,  # Solar diurnal
        }
        
        # Extreme event thresholds for Annapolis (in feet MLLW)
        self.flood_thresholds = {
            'minor_flood': 2.6,  # Minor flooding begins
            'moderate_flood': 3.2,  # Moderate flooding
            'major_flood': 4.2   # Major flooding
        }
    
    def load_data_from_db(self, table_name, hours_back=168):  # Default 7 days
        """Load data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        query = f"""
            SELECT * FROM {table_name} 
            WHERE timestamp >= '{start_time_str}'
            ORDER BY timestamp
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading {table_name}: {e}")
            conn.close()
            return pd.DataFrame()
    
    def clean_tide_data(self, df):
        """Clean and quality control tide data"""
        if df.empty:
            print("Tide data is empty - nothing to clean")
            return df
        
        print(f"Starting tide data cleaning with {len(df)} records")
        
        # Store non-numeric columns before processing
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Non-numeric columns: {non_numeric_cols}")
        
        # Handle quality flags - NOAA returns different formats
        if 'quality_flag' in df.columns:
            print(f"Quality flags present: {df['quality_flag'].value_counts().to_dict()}")
            
            # Check if quality flags are in comma-separated format (newer NOAA API)
            sample_flag = df['quality_flag'].iloc[0] if len(df) > 0 else None
            
            if sample_flag and ',' in str(sample_flag):
                # Handle comma-separated quality flags
                # For NOAA CO-OPS: first digit usually indicates data quality
                # 1 = verified, 0 = preliminary/unverified
                print("Detected comma-separated quality flags")
                
                # Extract first quality indicator
                df['quality_numeric'] = df['quality_flag'].apply(
                    lambda x: str(x).split(',')[0] if pd.notna(x) else '0'
                )
                
                # Accept both verified (1) and preliminary (0) data
                valid_quality = ['0', '1']  # Both preliminary and verified
                df = df[df['quality_numeric'].isin(valid_quality)].copy()
                
            else:
                # Handle traditional single-character quality flags
                print("Detected traditional quality flags")
                valid_flags = ['v', 'p', 'e', 't', 'n']  # verified, preliminary, estimated, provisional, null
                df = df[df['quality_flag'].isin(valid_flags)].copy()
            
            print(f"After quality filtering: {len(df)} records")
        
        if df.empty:
            print("WARNING: All data removed by quality filtering!")
            return df
        
        # Remove extreme outliers (beyond physical limits)
        if 'water_level' in df.columns:
            # Check for NaN values first
            nan_count = df['water_level'].isna().sum()
            if nan_count > 0:
                print(f"Found {nan_count} NaN values in water_level")
                df = df.dropna(subset=['water_level'])
                print(f"After removing NaN values: {len(df)} records")
            
            if df.empty:
                print("WARNING: All data removed due to NaN water levels!")
                return df
            
            print(f"Water level range before outlier removal: {df['water_level'].min():.3f} to {df['water_level'].max():.3f}")
            
            # For Annapolis, reasonable range is -2 to 8 feet MLLW
            df = df[(df['water_level'] >= -3) & (df['water_level'] <= 10)].copy()
            print(f"After physical limits filtering: {len(df)} records")
            
            if len(df) > 10:  # Only do statistical filtering if we have enough data
                # Statistical outlier removal using IQR method
                Q1 = df['water_level'].quantile(0.25)
                Q3 = df['water_level'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.5 * IQR  # More conservative than 1.5
                upper_bound = Q3 + 2.5 * IQR
                
                print(f"IQR outlier bounds: {lower_bound:.3f} to {upper_bound:.3f}")
                
                original_count = len(df)
                df = df[(df['water_level'] >= lower_bound) & 
                       (df['water_level'] <= upper_bound)].copy()
                removed = original_count - len(df)
                print(f"After statistical outlier removal: {len(df)} records ({removed} removed)")
        
        if df.empty:
            print("WARNING: All data removed during cleaning!")
            return df
        
        # Separate numeric and non-numeric columns for resampling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns for resampling: {numeric_cols}")
        
        if numeric_cols:
            try:
                # Resample numeric columns to 6-minute intervals
                df_numeric = df[numeric_cols].resample('6T').mean()
                print(f"After resampling numeric data: {len(df_numeric)} records")
                
                # Handle non-numeric columns (forward fill for categorical data)
                filtered_non_numeric = [col for col in non_numeric_cols if col in df.columns]
                if filtered_non_numeric:
                    df_non_numeric = df[filtered_non_numeric].resample('6T').first()
                    # Combine numeric and non-numeric data
                    df = pd.concat([df_numeric, df_non_numeric], axis=1)
                else:
                    df = df_numeric
                    
                print(f"Final cleaned tide data: {len(df)} records")
                
            except Exception as e:
                print(f"Error during resampling: {e}")
                return pd.DataFrame()
        else:
            print("No numeric columns found for resampling")
            # If no numeric columns, just resample with first() method
            df = df.resample('6T').first()
        
        # Interpolate short gaps (up to 1 hour) in numeric columns only
        for col in df.columns:
            if col not in non_numeric_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                before_interp = df[col].isna().sum()
                df[col] = df[col].interpolate(method='time', limit=10)
                after_interp = df[col].isna().sum()
                if before_interp > after_interp:
                    print(f"Interpolated {before_interp - after_interp} missing values in {col}")
        
        return df
    
    def clean_weather_data(self, df):
        """Clean weather data"""
        if df.empty:
            return df
        
        # Store non-numeric columns before resampling
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove unreasonable values
        if 'air_temperature' in df.columns:
            df = df[(df['air_temperature'] >= -20) & (df['air_temperature'] <= 120)].copy()
        
        if 'wind_speed' in df.columns:
            df = df[df['wind_speed'] >= 0].copy()
            df = df[df['wind_speed'] <= 150].copy()  # Remove hurricane-force+ winds unless real
        
        if 'barometric_pressure' in df.columns:
            df = df[(df['barometric_pressure'] >= 28.0) & 
                   (df['barometric_pressure'] <= 32.0)].copy()
        
        # Separate numeric and non-numeric columns for resampling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Resample numeric columns to 6-minute intervals
            df_numeric = df[numeric_cols].resample('6T').mean()
            
            # Handle non-numeric columns (forward fill for categorical data)
            if non_numeric_cols:
                df_non_numeric = df[non_numeric_cols].resample('6T').first()
                # Combine numeric and non-numeric data
                df = pd.concat([df_numeric, df_non_numeric], axis=1)
            else:
                df = df_numeric
        else:
            # If no numeric columns, just resample with first() method
            df = df.resample('6T').first()
        
        # Interpolate short gaps in numeric columns only
        for col in df.columns:
            if col not in non_numeric_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                df[col] = df[col].interpolate(method='time', limit=10)
        
        return df
    
    def clean_stream_data(self, df):
        """Clean USGS stream data"""
        if df.empty:
            return df
        
        # Store non-numeric columns before processing
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove negative discharge values
        if 'discharge' in df.columns:
            df = df[df['discharge'] >= 0].copy()
        
        # Remove unreasonable gage heights
        if 'gage_height' in df.columns:
            df = df[df['gage_height'] >= -10].copy()  # Allow for some negative values
        
        # Separate numeric and non-numeric columns for resampling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Resample numeric columns to 6-minute intervals
            df_numeric = df[numeric_cols].resample('6T').mean()
            
            # Handle non-numeric columns (forward fill for categorical data)
            if non_numeric_cols:
                df_non_numeric = df[non_numeric_cols].resample('6T').first()
                # Combine numeric and non-numeric data
                df = pd.concat([df_numeric, df_non_numeric], axis=1)
            else:
                df = df_numeric
        else:
            # If no numeric columns, just resample with first() method
            df = df.resample('6T').first()
        
        # Interpolate gaps in numeric columns only
        for col in df.columns:
            if col not in non_numeric_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                df[col] = df[col].interpolate(method='time', limit=20)
        
        return df
    
    def create_tidal_features(self, tide_df):
        """Extract tidal harmonic components and create tidal features"""
        if tide_df.empty:
            print("No tide data for feature creation")
            return tide_df
        
        print(f"Starting tidal feature creation with {len(tide_df)} records")
        print(f"Tide data columns: {list(tide_df.columns)}")
        
        if 'water_level' not in tide_df.columns:
            print("ERROR: No 'water_level' column found in tide data")
            return tide_df
        
        df = tide_df.copy()
        
        # Calculate water level anomaly (observed - predicted)
        if 'prediction' in df.columns:
            df['water_level_anomaly'] = df['water_level'] - df['prediction']
            print("Created water_level_anomaly using predictions")
        else:
            # If no predictions, use detrended anomaly
            try:
                from scipy import signal
                df['water_level_anomaly'] = signal.detrend(df['water_level'].fillna(method='ffill'))
                print("Created water_level_anomaly using detrending")
            except Exception as e:
                print(f"Error creating anomaly: {e}")
                df['water_level_anomaly'] = df['water_level'] - df['water_level'].mean()
                print("Created simple anomaly (value - mean)")
        
        # Create time-based tidal features
        timestamps = df.index
        
        if len(timestamps) == 0:
            print("No timestamps available for tidal features")
            return df
        
        # Convert to hours since start for calculations
        start_time = timestamps[0]
        hours_since_start = [(ts - start_time).total_seconds() / 3600 for ts in timestamps]
        
        # Extract major tidal constituents
        feature_count = 0
        for constituent, period in self.tidal_constituents.items():
            try:
                # Create sinusoidal components
                omega = 2 * np.pi / period  # Angular frequency
                df[f'tidal_{constituent}_cos'] = np.cos(omega * np.array(hours_since_start))
                df[f'tidal_{constituent}_sin'] = np.sin(omega * np.array(hours_since_start))
                feature_count += 2
            except Exception as e:
                print(f"Error creating tidal features for {constituent}: {e}")
        
        print(f"Created {feature_count} tidal harmonic features")
        
        # Tidal range and velocity features
        try:
            df['water_level_1h_change'] = df['water_level'].diff(10)  # 10 * 6min = 1 hour
            df['water_level_velocity'] = df['water_level'].diff()  # 6-minute rate of change
            df['water_level_acceleration'] = df['water_level_velocity'].diff()
            print("Created tidal velocity and acceleration features")
        except Exception as e:
            print(f"Error creating velocity features: {e}")
        
        # Rolling statistics
        try:
            df['water_level_1h_mean'] = df['water_level'].rolling(10).mean()
            df['water_level_1h_std'] = df['water_level'].rolling(10).std()
            df['water_level_6h_max'] = df['water_level'].rolling(60).max()
            df['water_level_6h_min'] = df['water_level'].rolling(60).min()
            df['tidal_range_6h'] = df['water_level_6h_max'] - df['water_level_6h_min']
            print("Created rolling statistics features")
        except Exception as e:
            print(f"Error creating rolling features: {e}")
        
        # Flood stage indicators
        flood_features = 0
        for stage, threshold in self.flood_thresholds.items():
            try:
                df[f'approaching_{stage}'] = (df['water_level'] > threshold - 0.5).astype(int)
                df[f'exceeds_{stage}'] = (df['water_level'] > threshold).astype(int)
                flood_features += 2
            except Exception as e:
                print(f"Error creating flood features for {stage}: {e}")
        
        print(f"Created {flood_features} flood indicator features")
        print(f"Total tidal features created: {len(df.columns) - len(tide_df.columns)}")
        
        return df
    
    def create_weather_features(self, weather_df):
        """Create weather-based features"""
        if weather_df.empty:
            return weather_df
        
        df = weather_df.copy()
        
        # Pressure features (important for storm surge)
        if 'barometric_pressure' in df.columns:
            df['pressure_1h_change'] = df['barometric_pressure'].diff(10)
            df['pressure_3h_change'] = df['barometric_pressure'].diff(30)
            df['pressure_6h_change'] = df['barometric_pressure'].diff(60)
            df['pressure_trend'] = df['barometric_pressure'].rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
            )
            
            # Pressure anomaly (deviation from local mean)
            df['pressure_anomaly'] = df['barometric_pressure'] - df['barometric_pressure'].rolling(720).mean()  # 3-day average
        
        # Wind features
        if 'wind_speed' in df.columns:
            df['wind_speed_1h_max'] = df['wind_speed'].rolling(10).max()
            df['wind_speed_6h_mean'] = df['wind_speed'].rolling(60).mean()
            
            # Wind stress (quadratic relationship with surge)
            df['wind_stress'] = df['wind_speed'] ** 2
            
            if 'wind_direction' in df.columns:
                # Convert wind direction to components
                wind_dir_rad = np.radians(df['wind_direction'])
                df['wind_u'] = df['wind_speed'] * np.cos(wind_dir_rad)  # East-west component
                df['wind_v'] = df['wind_speed'] * np.sin(wind_dir_rad)  # North-south component
                
                # Onshore/offshore wind components for Annapolis (facing southeast)
                # Annapolis waterfront faces approximately 140 degrees
                shore_normal = np.radians(140)  # Normal to shore
                df['wind_onshore'] = df['wind_speed'] * np.cos(wind_dir_rad - shore_normal)
                df['wind_alongshore'] = df['wind_speed'] * np.sin(wind_dir_rad - shore_normal)
        
        # Temperature features
        if 'air_temperature' in df.columns:
            df['temp_1h_change'] = df['air_temperature'].diff(10)
            df['temp_6h_range'] = df['air_temperature'].rolling(60).max() - df['air_temperature'].rolling(60).min()
        
        return df
    
    def create_stream_features(self, stream_df):
        """Create stream flow features"""
        if stream_df.empty:
            return stream_df
        
        df = stream_df.copy()
        
        if 'discharge' in df.columns:
            # Flow velocity indicators
            df['discharge_1h_change'] = df['discharge'].diff(10)
            df['discharge_6h_change'] = df['discharge'].diff(60)
            df['discharge_1h_max'] = df['discharge'].rolling(10).max()
            df['discharge_6h_mean'] = df['discharge'].rolling(60).mean()
            
            # Flow anomaly
            df['discharge_anomaly'] = df['discharge'] - df['discharge'].rolling(1440).mean()  # 6-day average
            
            # Baseflow separation (simple digital filter)
            # This separates quick flow (from precipitation) from baseflow
            alpha = 0.925  # Filter parameter
            df['baseflow'] = np.nan
            df['quickflow'] = np.nan
            
            if len(df) > 1:
                baseflow = [df['discharge'].iloc[0]]
                for i in range(1, len(df)):
                    bt = alpha * baseflow[i-1] + (1 + alpha) / 2 * (df['discharge'].iloc[i] - df['discharge'].iloc[i-1])
                    bt = min(bt, df['discharge'].iloc[i])
                    baseflow.append(bt)
                
                df['baseflow'] = baseflow
                df['quickflow'] = df['discharge'] - df['baseflow']
        
        if 'gage_height' in df.columns:
            df['gage_height_1h_change'] = df['gage_height'].diff(10)
            df['gage_height_velocity'] = df['gage_height'].diff()
        
        return df
    
    def create_compound_features(self, tide_df, weather_df, stream_df):
        """Create interaction features between different data sources"""
        # Start with tide data as base (most complete temporal coverage)
        if tide_df.empty:
            return pd.DataFrame()
        
        combined_df = tide_df.copy()
        
        # Merge weather data
        if not weather_df.empty:
            # Align timestamps and merge
            weather_resampled = weather_df.reindex(combined_df.index, method='nearest', tolerance='30T')
            for col in weather_resampled.columns:
                if col not in combined_df.columns:
                    combined_df[col] = weather_resampled[col]
        
        # Merge stream data
        if not stream_df.empty:
            stream_resampled = stream_df.reindex(combined_df.index, method='nearest', tolerance='30T')
            for col in stream_resampled.columns:
                if col not in combined_df.columns:
                    combined_df[col] = stream_resampled[col]
        
        # Create interaction features
        if all(col in combined_df.columns for col in ['water_level_anomaly', 'barometric_pressure']):
            combined_df['surge_pressure_interaction'] = combined_df['water_level_anomaly'] * combined_df['pressure_anomaly']
        
        if all(col in combined_df.columns for col in ['water_level_anomaly', 'wind_onshore']):
            combined_df['surge_wind_interaction'] = combined_df['water_level_anomaly'] * combined_df['wind_onshore']
        
        if all(col in combined_df.columns for col in ['discharge_anomaly', 'water_level_anomaly']):
            combined_df['compound_flood_risk'] = (
                (combined_df['discharge_anomaly'] > 0) & 
                (combined_df['water_level_anomaly'] > 0)
            ).astype(int)
        
        # Time-based features
        combined_df['hour'] = combined_df.index.hour
        combined_df['day_of_year'] = combined_df.index.dayofyear
        combined_df['month'] = combined_df.index.month
        combined_df['day_of_week'] = combined_df.index.dayofweek
        
        # Seasonal features
        combined_df['season_cos'] = np.cos(2 * np.pi * combined_df.index.dayofyear / 365.25)
        combined_df['season_sin'] = np.sin(2 * np.pi * combined_df.index.dayofyear / 365.25)
        
        # Storm season indicator (hurricane season: June 1 - November 30)
        combined_df['hurricane_season'] = (
            (combined_df['month'] >= 6) & (combined_df['month'] <= 11)
        ).astype(int)
        
        return combined_df
    
    def create_extreme_event_labels(self, df):
        """Create labels for extreme water level events"""
        if df.empty or 'water_level' not in df.columns:
            return df
        
        # Create binary labels for different flood stages
        for stage, threshold in self.flood_thresholds.items():
            df[f'label_{stage}'] = (df['water_level'] > threshold).astype(int)
        
        # Create lead time labels (predict flooding X hours in advance)
        lead_times = [1, 3, 6, 12, 24]  # hours
        for hours in lead_times:
            periods = hours * 10  # 6-minute intervals
            for stage in self.flood_thresholds.keys():
                df[f'label_{stage}_{hours}h_lead'] = df[f'label_{stage}'].shift(-periods)
        
        # Create continuous regression targets
        df['water_level_1h_ahead'] = df['water_level'].shift(-10)
        df['water_level_3h_ahead'] = df['water_level'].shift(-30)
        df['water_level_6h_ahead'] = df['water_level'].shift(-60)
        df['water_level_12h_ahead'] = df['water_level'].shift(-120)
        
        # Maximum water level in next N hours
        df['max_water_level_6h'] = df['water_level'].rolling(60).max().shift(-60)
        df['max_water_level_12h'] = df['water_level'].rolling(120).max().shift(-120)
        
        return df
    
    def process_all_data(self, hours_back=168):
        """Main processing pipeline"""
        print("Loading data from database...")
        
        # Load raw data
        tide_data = self.load_data_from_db('tide_data', hours_back)
        weather_data = self.load_data_from_db('weather_data', hours_back)
        stream_data = self.load_data_from_db('stream_data', hours_back)
        
        print(f"Loaded {len(tide_data)} tide records, {len(weather_data)} weather records, {len(stream_data)} stream records")
        
        # Debug: Check raw data columns and samples
        if not tide_data.empty:
            print(f"Tide data columns: {list(tide_data.columns)}")
            print(f"Tide data sample:\n{tide_data.head()}")
        else:
            print("No tide data loaded!")
            
        if not weather_data.empty:
            print(f"Weather data columns: {list(weather_data.columns)}")
            print(f"Weather data sample:\n{weather_data.head()}")
        
        # Clean data
        print("Cleaning data...")
        tide_clean = self.clean_tide_data(tide_data)
        weather_clean = self.clean_weather_data(weather_data)
        stream_clean = self.clean_stream_data(stream_data)
        
        print(f"After cleaning: tide={len(tide_clean)}, weather={len(weather_clean)}, stream={len(stream_clean)}")
        
        # Check if we have any data after cleaning
        if tide_clean.empty:
            print("ERROR: No tide data after cleaning!")
            return pd.DataFrame()
        
        # Create features
        print("Creating features...")
        tide_features = self.create_tidal_features(tide_clean)
        weather_features = self.create_weather_features(weather_clean)
        stream_features = self.create_stream_features(stream_clean)
        
        print(f"After feature creation: tide={len(tide_features)}, weather={len(weather_features)}, stream={len(stream_features)}")
        
        # Combine all features
        print("Combining features...")
        combined_features = self.create_compound_features(tide_features, weather_features, stream_features)
        
        print(f"Combined features shape: {combined_features.shape}")
        if not combined_features.empty:
            print(f"Combined features columns: {list(combined_features.columns)[:10]}...")  # Show first 10 columns
        
        # Create prediction labels
        print("Creating prediction labels...")
        final_df = self.create_extreme_event_labels(combined_features)
        
        print(f"Final dataset shape: {final_df.shape}")
        if not final_df.empty:
            print(f"Features created: {list(final_df.columns)[:15]}...")  # Show first 15 columns
        else:
            print("WARNING: Final dataset is empty!")
        
        return final_df
    
    def save_processed_data(self, df, filename='processed_annapolis_data.csv'):
        """Save processed data to CSV"""
        df.to_csv(filename)
        print(f"Processed data saved to {filename}")
    
    def generate_data_summary(self, df):
        """Generate summary statistics and visualizations"""
        if df.empty:
            print("No data to summarize")
            return
        
        print("\n=== DATA SUMMARY ===")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Total records: {len(df)}")
        print(f"Data completeness:")
        
        # Calculate completeness for key variables
        key_vars = ['water_level', 'barometric_pressure', 'wind_speed', 'discharge']
        for var in key_vars:
            if var in df.columns:
                completeness = (1 - df[var].isna().sum() / len(df)) * 100
                print(f"  {var}: {completeness:.1f}%")
        
        # Summary statistics
        print(f"\nSummary statistics for key variables:")
        if 'water_level' in df.columns:
            print(f"Water level (feet MLLW):")
            print(f"  Mean: {df['water_level'].mean():.2f}")
            print(f"  Std: {df['water_level'].std():.2f}")
            print(f"  Min: {df['water_level'].min():.2f}")
            print(f"  Max: {df['water_level'].max():.2f}")
            
            # Flood exceedances
            for stage, threshold in self.flood_thresholds.items():
                exceedances = (df['water_level'] > threshold).sum()
                pct = (exceedances / len(df)) * 100
                print(f"  {stage} flood exceedances: {exceedances} ({pct:.2f}%)")
        
        # Create visualizations
        self.create_summary_plots(df)
    
    def create_summary_plots(self, df):
        """Create summary plots of the processed data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Water level time series
        if 'water_level' in df.columns:
            axes[0,0].plot(df.index, df['water_level'], alpha=0.7, linewidth=0.5)
            
            # Add flood thresholds
            for stage, threshold in self.flood_thresholds.items():
                axes[0,0].axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                                label=f'{stage}: {threshold} ft')
            
            axes[0,0].set_title('Water Level Time Series')
            axes[0,0].set_ylabel('Water Level (ft MLLW)')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Water level distribution
        if 'water_level' in df.columns:
            axes[0,1].hist(df['water_level'].dropna(), bins=50, alpha=0.7, density=True)
            axes[0,1].set_title('Water Level Distribution')
            axes[0,1].set_xlabel('Water Level (ft MLLW)')
            axes[0,1].set_ylabel('Density')
            axes[0,1].grid(True, alpha=0.3)
        
        # Weather variables
        if 'barometric_pressure' in df.columns:
            ax2 = axes[1,0]
            ax2.plot(df.index, df['barometric_pressure'], color='blue', alpha=0.7, linewidth=0.5)
            ax2.set_ylabel('Pressure (inHg)', color='blue')
            ax2.set_title('Weather Variables')
            
            if 'wind_speed' in df.columns:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(df.index, df['wind_speed'], color='red', alpha=0.7, linewidth=0.5)
                ax2_twin.set_ylabel('Wind Speed (mph)', color='red')
            
            ax2.grid(True, alpha=0.3)
        
        # Stream discharge
        if 'discharge' in df.columns:
            axes[1,1].plot(df.index, df['discharge'], alpha=0.7, linewidth=0.5, color='green')
            axes[1,1].set_title('Severn River Discharge')
            axes[1,1].set_ylabel('Discharge (cfs)')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('annapolis_data_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 5:
            plt.figure(figsize=(12, 10))
            # Select key variables for correlation plot
            key_vars = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['water_level', 'pressure', 'wind', 'discharge', 'anomaly'])][:15]
            
            if len(key_vars) > 2:
                corr_matrix = df[key_vars].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.savefig('annapolis_feature_correlations.png', dpi=300, bbox_inches='tight')
                plt.show()

if __name__ == "__main__":
    # Initialize processor
    processor = AnnapolisDataProcessor()
    
    # Process all data (default: last 7 days)
    processed_data = processor.process_all_data(hours_back=168)
    
    # Generate summary
    processor.generate_data_summary(processed_data)
    
    # Save processed data
    processor.save_processed_data(processed_data)
    
    print("Data processing complete!")