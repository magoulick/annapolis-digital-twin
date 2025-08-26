# ====================================================================
# STANDALONE EXTENDED-RANGE COASTAL FLOOD PREDICTION SYSTEM
# No Docker Required - Pure Python Implementation
# ====================================================================

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    import xgboost as xgb
    print("✓ Core ML libraries loaded successfully")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Install with: pip install scikit-learn xgboost")
    sys.exit(1)

# Optional deep learning (system works without it)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print("✓ TensorFlow available for advanced models")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available - will use tree-based models only")

# Optional web interface
try:
    import flask
    from flask import Flask, request, jsonify, render_template_string
    FLASK_AVAILABLE = True
    print("✓ Flask available for web interface")
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠ Flask not available - will run in command-line mode only")

# Optional data fetching
try:
    import requests
    REQUESTS_AVAILABLE = True
    print("✓ Requests available for real-time data")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠ Requests not available - will use synthetic data")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================================================================
# SIMPLIFIED DATA MANAGEMENT
# ====================================================================

class SimpleDataManager:
    """Simplified data management using SQLite and file storage"""
    
    def __init__(self, data_dir: str = "./extended_range_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # SQLite database for storing data
        self.db_path = self.data_dir / "extended_range.db"
        self.init_database()
        
        # File-based storage for models and results
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Data directory initialized: {self.data_dir}")
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_data (
                timestamp TEXT PRIMARY KEY,
                station_id TEXT,
                water_level REAL,
                air_pressure REAL,
                wind_speed REAL,
                wind_direction REAL,
                air_temperature REAL,
                data_source TEXT DEFAULT 'synthetic'
            )
        ''')
        
        # Climate indices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS climate_indices (
                timestamp TEXT,
                index_name TEXT,
                value REAL,
                PRIMARY KEY (timestamp, index_name)
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                lead_time_hours INTEGER,
                prediction REAL,
                uncertainty REAL,
                tier TEXT,
                model_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def generate_synthetic_data(self, start_date: str = "2019-01-01", 
                               end_date: str = "2024-12-31", 
                               station_id: str = "8575512"):
        """Generate realistic synthetic training data"""
        
        logger.info("Generating synthetic training data...")
        
        # Create 6-minute intervals
        date_range = pd.date_range(start_date, end_date, freq='6T')
        hours = np.array([(dt - pd.to_datetime(start_date)).total_seconds() / 3600 
                         for dt in date_range])
        
        # Realistic tidal constituents for Chesapeake Bay
        M2 = 1.2 * np.sin(2 * np.pi * hours / 12.42)  # Principal lunar semi-diurnal
        S2 = 0.4 * np.sin(2 * np.pi * hours / 12.00)  # Principal solar semi-diurnal
        O1 = 0.3 * np.sin(2 * np.pi * hours / 25.82)  # Lunar diurnal
        
        # Add sea level rise trend (3mm/year = 0.01 ft/year)
        trend = 0.01 * hours / (24 * 365)
        
        # Add weather noise and seasonal patterns
        noise = np.random.normal(0, 0.1, len(hours))
        seasonal = 0.2 * np.sin(2 * np.pi * hours / (24 * 365.25))
        
        water_level = M2 + S2 + O1 + trend + seasonal + noise + 2.0  # Add mean level
        
        # Meteorological data
        air_pressure = 1013.25 + 10 * np.sin(2 * np.pi * hours / (24 * 7)) + np.random.normal(0, 5, len(hours))
        wind_speed = np.abs(15 + 10 * np.sin(2 * np.pi * hours / (24 * 3)) + np.random.normal(0, 3, len(hours)))
        wind_direction = 180 + 60 * np.sin(2 * np.pi * hours / (24 * 2)) + np.random.normal(0, 20, len(hours))
        air_temperature = 15 + 10 * np.sin(2 * np.pi * (hours / (24 * 365) - 0.25)) + np.random.normal(0, 2, len(hours))
        
        # Add extreme events (storms every ~90 days)
        storm_dates = pd.date_range(start_date, end_date, freq='90D')
        for storm_date in storm_dates:
            storm_mask = (date_range >= storm_date) & (date_range <= storm_date + timedelta(days=2))
            storm_indices = np.where(storm_mask)[0]
            if len(storm_indices) > 0:
                water_level[storm_indices] += 0.5 + 0.3 * np.random.random(len(storm_indices))
                wind_speed[storm_indices] += 20 + 10 * np.random.random(len(storm_indices))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range.strftime('%Y-%m-%d %H:%M:%S'),
            'station_id': station_id,
            'water_level': water_level,
            'air_pressure': air_pressure,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'air_temperature': air_temperature,
            'data_source': 'synthetic'
        })
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        df.to_sql('historical_data', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"Generated {len(df)} synthetic records from {start_date} to {end_date}")
        return df
    
    def generate_synthetic_climate_data(self):
        """Generate synthetic climate indices"""
        
        logger.info("Generating synthetic climate indices...")
        
        # Monthly data from 2019-2024
        monthly_dates = pd.date_range('2019-01-15', '2024-12-15', freq='MS')
        
        climate_data = []
        
        # ENSO (Nino 3.4) - realistic persistence
        enso_value = 0.0
        for date in monthly_dates:
            # Add persistence and random variation
            change = np.random.normal(0, 0.3)
            enso_value = 0.7 * enso_value + change
            enso_value = np.clip(enso_value, -3.0, 3.0)
            
            climate_data.append({
                'timestamp': date.strftime('%Y-%m-%d'),
                'index_name': 'NINO34',
                'value': enso_value
            })
        
        # NAO (North Atlantic Oscillation)
        for date in monthly_dates:
            nao_value = np.random.normal(0, 1)
            climate_data.append({
                'timestamp': date.strftime('%Y-%m-%d'),
                'index_name': 'NAO',
                'value': nao_value
            })
        
        # PDO (Pacific Decadal Oscillation)
        for date in monthly_dates:
            pdo_value = np.random.normal(0, 0.8)
            climate_data.append({
                'timestamp': date.strftime('%Y-%m-%d'),
                'index_name': 'PDO',
                'value': pdo_value
            })
        
        # Store in database
        climate_df = pd.DataFrame(climate_data)
        conn = sqlite3.connect(self.db_path)
        climate_df.to_sql('climate_indices', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"Generated {len(climate_df)} climate index records")
        return climate_df
    
    def load_historical_data(self, station_id: str = "8575512") -> pd.DataFrame:
        """Load historical data from database"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Check if data exists
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM historical_data WHERE station_id = ?", (station_id,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("No historical data found, generating synthetic data...")
            self.generate_synthetic_data()
            self.generate_synthetic_climate_data()
        
        # Load data
        df = pd.read_sql_query(
            "SELECT * FROM historical_data WHERE station_id = ? ORDER BY timestamp",
            conn, params=(station_id,)
        )
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            logger.info(f"Loaded {len(df)} historical records for station {station_id}")
        
        return df
    
    def load_climate_data(self) -> Dict[str, pd.DataFrame]:
        """Load climate indices from database"""
        
        conn = sqlite3.connect(self.db_path)
        
        climate_data = {}
        for index_name in ['NINO34', 'NAO', 'PDO']:
            df = pd.read_sql_query(
                "SELECT timestamp, value FROM climate_indices WHERE index_name = ? ORDER BY timestamp",
                conn, params=(index_name,)
            )
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                climate_data[index_name] = df
        
        conn.close()
        
        logger.info(f"Loaded climate indices: {list(climate_data.keys())}")
        return climate_data

# ====================================================================
# SIMPLIFIED FEATURE ENGINEERING
# ====================================================================

class SimpleFeatureEngine:
    """Simplified feature engineering for extended-range prediction"""
    
    def __init__(self):
        # Core tidal constituents
        self.tidal_constituents = {
            'M2': 12.42,   # Principal lunar semi-diurnal
            'S2': 12.00,   # Principal solar semi-diurnal
            'O1': 25.82,   # Lunar diurnal
            'K1': 23.93    # Lunar diurnal
        }
    
    def create_features(self, data_df: pd.DataFrame, climate_data: Dict) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        logger.info("Creating extended-range features...")
        
        features_df = data_df.copy()
        
        # Core water level persistence features (98.9% importance from your current system)
        features_df['max_water_level_24h'] = features_df['water_level'].rolling(window=240, min_periods=1).max()
        features_df['max_water_level_12h'] = features_df['water_level'].rolling(window=120, min_periods=1).max()
        features_df['max_water_level_48h'] = features_df['water_level'].rolling(window=480, min_periods=1).max()
        features_df['mean_water_level_24h'] = features_df['water_level'].rolling(window=240, min_periods=1).mean()
        features_df['std_water_level_24h'] = features_df['water_level'].rolling(window=240, min_periods=1).std()
        
        # Tidal velocity and acceleration
        features_df['water_level_velocity'] = features_df['water_level'].diff()
        features_df['water_level_acceleration'] = features_df['water_level_velocity'].diff()
        
        # Tidal harmonic features
        self._add_tidal_harmonics(features_df)
        
        # Meteorological features
        self._add_meteorological_features(features_df)
        
        # Climate features
        self._add_climate_features(features_df, climate_data)
        
        # Seasonal and temporal features
        self._add_temporal_features(features_df)
        
        # Pattern persistence features
        self._add_pattern_features(features_df)
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Created {features_df.shape[1]} features")
        return features_df
    
    def _add_tidal_harmonics(self, df: pd.DataFrame):
        """Add tidal harmonic components"""
        
        if df.empty:
            return
        
        start_time = df.index[0]
        hours_since_start = [(ts - start_time).total_seconds() / 3600 for ts in df.index]
        
        for constituent, period in self.tidal_constituents.items():
            omega = 2 * np.pi / period
            df[f'tidal_{constituent}_cos'] = np.cos(omega * np.array(hours_since_start))
            df[f'tidal_{constituent}_sin'] = np.sin(omega * np.array(hours_since_start))
    
    def _add_meteorological_features(self, df: pd.DataFrame):
        """Add meteorological features"""
        
        if 'air_pressure' in df.columns:
            df['pressure_anomaly'] = df['air_pressure'] - df['air_pressure'].rolling(720, min_periods=1).mean()
            df['pressure_tendency_6h'] = df['air_pressure'].diff(60)
        
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            # Wind components
            df['wind_u'] = -df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
            df['wind_v'] = -df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
            
            # Onshore wind component (for Chesapeake Bay)
            bay_orientation = 15.0  # degrees from north
            df['wind_onshore'] = (df['wind_u'] * np.sin(np.radians(bay_orientation)) + 
                                 df['wind_v'] * np.cos(np.radians(bay_orientation)))
            
            # Wind stress
            df['wind_stress'] = 0.001 * df['wind_speed']**2
    
    def _add_climate_features(self, df: pd.DataFrame, climate_data: Dict):
        """Add climate teleconnection features"""
        
        for index_name, climate_df in climate_data.items():
            if climate_df.empty:
                continue
            
            # Resample to match data frequency
            climate_resampled = climate_df.resample('6T').ffill()
            
            # Current value
            col_name = f'{index_name.lower()}_current'
            df[col_name] = climate_resampled['value'].reindex(df.index, method='ffill')
            
            # Lagged values (30, 60, 90 days)
            for lag_days in [30, 60, 90]:
                lag_periods = lag_days * 24 * 10  # Convert to 6-minute periods
                lag_col = f'{index_name.lower()}_lag_{lag_days}d'
                df[lag_col] = df[col_name].shift(lag_periods)
            
            # Climate state indicators
            if index_name == 'NINO34':
                df['enso_el_nino'] = (df[col_name] > 0.5).astype(int)
                df['enso_la_nina'] = (df[col_name] < -0.5).astype(int)
            elif index_name == 'NAO':
                df['nao_positive'] = (df[col_name] > 0).astype(int)
    
    def _add_temporal_features(self, df: pd.DataFrame):
        """Add seasonal and temporal features"""
        
        df['day_of_year'] = df.index.dayofyear
        df['hour_of_day'] = df.index.hour
        df['month'] = df.index.month
        
        # Seasonal cycles
        df['season_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['season_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Daily cycles
        df['daily_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['daily_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        
        # Storm season
        df['hurricane_season'] = ((df.index.month >= 6) & (df.index.month <= 11)).astype(int)
    
    def _add_pattern_features(self, df: pd.DataFrame):
        """Add pattern persistence features"""
        
        # Pressure pattern persistence
        if 'air_pressure' in df.columns:
            for window_hours in [24, 48, 72]:
                window_periods = window_hours * 10
                col_name = f'pressure_persistence_{window_hours}h'
                df[col_name] = df['air_pressure'].rolling(window_periods, min_periods=10).apply(
                    lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
                )
        
        # Water level pattern persistence
        for window_hours in [12, 24, 48]:
            window_periods = window_hours * 10
            col_name = f'water_level_persistence_{window_hours}h'
            df[col_name] = df['water_level'].rolling(window_periods, min_periods=10).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
            )

# ====================================================================
# SIMPLIFIED HIERARCHICAL MODELS
# ====================================================================

class SimpleHierarchicalEnsemble:
    """Simplified hierarchical ensemble for different prediction horizons"""
    
    def __init__(self):
        self.tier1_models = {}  # 1-3 days
        self.tier2_models = {}  # 3-7 days
        self.tier3_models = {}  # 7+ days
        self.scalers = {}
        
        # Performance targets
        self.performance_targets = {
            'tier1': 0.90,  # 1-3 days: R² >0.90
            'tier2': 0.75,  # 3-7 days: R² >0.75
            'tier3': 0.60   # 7+ days: R² >0.60
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models for each tier"""
        
        # Tier 1: 1-3 days (maintain current performance)
        self.tier1_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                learning_rate=0.01, n_estimators=1000, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, 
                random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
        }
        
        # Tier 2: 3-7 days (weather pattern recognition)
        self.tier2_models = {
            'random_forest_extended': RandomForestRegressor(
                n_estimators=150, max_depth=15, min_samples_split=8,
                random_state=42, n_jobs=-1
            ),
            'xgboost_extended': xgb.XGBRegressor(
                learning_rate=0.005, n_estimators=1500, max_depth=5,
                subsample=0.9, random_state=42, n_jobs=-1
            ),
            'gaussian_process': GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
                alpha=1e-10, random_state=42
            )
        }
        
        # Tier 3: 7+ days (climate-informed)
        self.tier3_models = {
            'random_forest_climate': RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_split=10,
                random_state=42, n_jobs=-1
            ),
            'xgboost_climate': xgb.XGBRegressor(
                learning_rate=0.003, n_estimators=2000, max_depth=4,
                subsample=0.95, random_state=42, n_jobs=-1
            )
        }
        
        # Add LSTM if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            logger.info("TensorFlow available - will add LSTM models during training")
    
    def get_tier_from_lead_time(self, lead_time_hours: int) -> str:
        """Determine tier based on lead time"""
        if lead_time_hours <= 72:
            return 'tier1'
        elif lead_time_hours <= 168:
            return 'tier2'
        else:
            return 'tier3'
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    lead_time_hours: int) -> Dict:
        """Train models for specific lead time"""
        
        tier = self.get_tier_from_lead_time(lead_time_hours)
        logger.info(f"Training {tier} models for {lead_time_hours}h lead time")
        
        # Get models for this tier
        if tier == 'tier1':
            models_to_train = self.tier1_models
        elif tier == 'tier2':
            models_to_train = self.tier2_models
        else:
            models_to_train = self.tier3_models
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[f'{tier}_{lead_time_hours}'] = scaler
        
        trained_models = {}
        
        for model_name, model in models_to_train.items():
            try:
                logger.info(f"  Training {model_name}...")
                
                # Handle Gaussian Process memory limitations
                if 'gaussian_process' in model_name and len(X_scaled) > 2000:
                    # Sample for GP training
                    indices = np.random.choice(len(X_scaled), 2000, replace=False)
                    model.fit(X_scaled[indices], y_train.iloc[indices])
                else:
                    model.fit(X_scaled, y_train)
                
                trained_models[model_name] = model
                logger.info(f"  ✓ {model_name} trained successfully")
                
            except Exception as e:
                logger.error(f"  ✗ {model_name} training failed: {e}")
                continue
        
        # Add LSTM if available and enough data
        if TENSORFLOW_AVAILABLE and len(X_scaled) > 500:
            try:
                lstm_model = self._train_lstm(X_scaled, y_train.values, tier)
                if lstm_model is not None:
                    trained_models['lstm'] = lstm_model
                    logger.info("  ✓ LSTM trained successfully")
            except Exception as e:
                logger.error(f"  ✗ LSTM training failed: {e}")
        
        # Store trained models
        if tier == 'tier1':
            self.tier1_models = trained_models
        elif tier == 'tier2':
            self.tier2_models = trained_models
        else:
            self.tier3_models = trained_models
        
        return trained_models
    
    def _train_lstm(self, X_scaled: np.ndarray, y_values: np.ndarray, tier: str):
        """Train LSTM model"""
        
        # Create sequences for LSTM
        if tier == 'tier1':
            sequence_length = 24  # 4 hours
        elif tier == 'tier2':
            sequence_length = 72  # 12 hours
        else:
            sequence_length = 120  # 20 hours
        
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_values, sequence_length)
        
        if len(X_sequences) < 100:
            return None
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_scaled.shape[1])),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_sequences, y_sequences,
            epochs=50, batch_size=32, validation_split=0.2,
            callbacks=[early_stopping], verbose=0
        )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        """Create sequences for LSTM training"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def predict(self, X_features: pd.DataFrame, lead_time_hours: int) -> Dict:
        """Make prediction using appropriate tier models"""
        
        tier = self.get_tier_from_lead_time(lead_time_hours)
        
        # Get models for this tier
        if tier == 'tier1':
            models = self.tier1_models
        elif tier == 'tier2':
            models = self.tier2_models
        else:
            models = self.tier3_models
        
        # Scale features
        scaler_key = f'{tier}_{lead_time_hours}'
        if scaler_key not in self.scalers:
            # Use tier default scaler
            scaler_key = f'{tier}_default'
            if scaler_key not in self.scalers:
                return {'error': f'No scaler found for {tier}'}
        
        X_scaled = self.scalers[scaler_key].transform(X_features)
        
        predictions = []
        model_names = []
        
        for model_name, model in models.items():
            try:
                if 'lstm' in model_name:
                    # Handle LSTM prediction
                    if tier == 'tier1':
                        seq_len = 24
                    elif tier == 'tier2':
                        seq_len = 72
                    else:
                        seq_len = 120
                    
                    if len(X_scaled) >= seq_len:
                        X_seq = X_scaled[-seq_len:].reshape(1, seq_len, -1)
                        pred = model.predict(X_seq, verbose=0)[0][0]
                        predictions.append(pred)
                        model_names.append(model_name)
                else:
                    # Standard ML model prediction
                    pred = model.predict(X_scaled[-1:])
                    predictions.append(pred[0])
                    model_names.append(model_name)
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                continue
        
        if not predictions:
            return {'error': 'No valid predictions generated'}
        
        # Ensemble prediction (simple average)
        ensemble_prediction = np.mean(predictions)
        prediction_std = np.std(predictions) if len(predictions) > 1 else 0.1
        
        # Uncertainty based on tier and ensemble spread
        uncertainty_multiplier = {'tier1': 1.0, 'tier2': 1.5, 'tier3': 2.0}
        uncertainty = prediction_std * uncertainty_multiplier[tier]
        
        # Confidence based on uncertainty
        confidence = max(0.3, 1.0 - uncertainty / max(abs(ensemble_prediction), 1.0))
        
        return {
            'prediction': float(ensemble_prediction),
            'uncertainty': float(uncertainty),
            'confidence': float(confidence),
            'tier': tier,
            'individual_predictions': dict(zip(model_names, predictions)),
            'n_models': len(predictions)
        }

# ====================================================================
# SIMPLIFIED CORRECTION SYSTEM
# ====================================================================

class SimpleCorrector:
    """Simplified adaptive correction system"""
    
    def __init__(self):
        self.base_correction = 0.87  # Your proven Chesapeake Bay factor
        self.seasonal_corrections = {
            'winter': 1.05,  # 5% increase for winter storms
            'spring': 1.02,  # 2% increase for spring tides
            'summer': 0.98,  # 2% decrease for summer
            'fall': 1.08     # 8% increase for fall hurricanes
        }
        self.climate_corrections = {
            'el_nino': 1.08,     # 8% increase during El Niño
            'la_nina': 0.94,     # 6% decrease during La Niña
            'nao_positive': 0.96, # 4% decrease when NAO positive
            'nao_negative': 1.12  # 12% increase when NAO negative
        }
    
    def apply_correction(self, prediction: float, timestamp: datetime, 
                        climate_state: Dict, lead_time_hours: int) -> float:
        """Apply adaptive corrections to prediction"""
        
        corrected = prediction * self.base_correction
        
        # Seasonal correction
        season = self._get_season(timestamp)
        if season in self.seasonal_corrections:
            corrected *= self.seasonal_corrections[season]
        
        # Climate corrections
        if climate_state.get('nino34_current', 0) > 0.5:
            corrected *= self.climate_corrections['el_nino']
        elif climate_state.get('nino34_current', 0) < -0.5:
            corrected *= self.climate_corrections['la_nina']
        
        if climate_state.get('nao_current', 0) > 0:
            corrected *= self.climate_corrections['nao_positive']
        elif climate_state.get('nao_current', 0) < 0:
            corrected *= self.climate_corrections['nao_negative']
        
        # Reduce correction strength for longer lead times
        if lead_time_hours > 24:
            correction_strength = max(0.1, 1.0 - (lead_time_hours - 24) / (7 * 24))
            final_correction = 1.0 + correction_strength * (corrected / prediction - 1.0)
            corrected = prediction * final_correction
        
        return corrected
    
    def _get_season(self, timestamp: datetime) -> str:
        """Get season from timestamp"""
        month = timestamp.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

# ====================================================================
# MAIN EXTENDED-RANGE SYSTEM
# ====================================================================

class StandaloneExtendedRangeSystem:
    """Complete standalone extended-range prediction system"""
    
    def __init__(self, data_dir: str = "./extended_range_data"):
        self.data_manager = SimpleDataManager(data_dir)
        self.feature_engine = SimpleFeatureEngine()
        self.ensemble = SimpleHierarchicalEnsemble()
        self.corrector = SimpleCorrector()
        
        # System state
        self.is_trained = False
        self.training_results = {}
        self.performance_metrics = {}
        
        logger.info("Standalone Extended-Range System initialized")
    
    def train_system(self, station_id: str = "8575512", 
                    lead_times: List[int] = None) -> Dict:
        """Train the complete system"""
        
        if lead_times is None:
            lead_times = [24, 48, 72, 120, 168, 240, 336]  # 1, 2, 3, 5, 7, 10, 14 days
        
        logger.info("=" * 50)
        logger.info("TRAINING EXTENDED-RANGE SYSTEM")
        logger.info("=" * 50)
        
        # Step 1: Load data
        logger.info("Step 1: Loading training data...")
        historical_data = self.data_manager.load_historical_data(station_id)
        climate_data = self.data_manager.load_climate_data()
        
        if historical_data.empty:
            logger.error("No historical data available")
            return {'error': 'No training data'}
        
        # Step 2: Create features
        logger.info("Step 2: Creating features...")
        features_df = self.feature_engine.create_features(historical_data, climate_data)
        
        # Step 3: Train models for each lead time
        logger.info("Step 3: Training models...")
        
        training_results = {}
        performance_summary = {}
        
        # Prepare data splits
        feature_cols = [col for col in features_df.columns if col != 'water_level']
        n_total = len(features_df)
        n_train = int(n_total * 0.8)  # 80% for training
        
        X_train = features_df[feature_cols].iloc[:n_train]
        y_train = features_df['water_level'].iloc[:n_train]
        X_val = features_df[feature_cols].iloc[n_train:]
        y_val = features_df['water_level'].iloc[n_train:]
        
        logger.info(f"Training data: {len(X_train)} samples")
        logger.info(f"Validation data: {len(X_val)} samples")
        
        for lead_time_hours in lead_times:
            logger.info(f"\nTraining for {lead_time_hours}h lead time...")
            
            # Create shifted target
            y_shifted = y_train.shift(-lead_time_hours // 6)  # Convert to 6-min periods
            valid_idx = y_shifted.dropna().index
            
            if len(valid_idx) < 100:
                logger.warning(f"Insufficient data for {lead_time_hours}h lead time")
                continue
            
            X_lead = X_train.loc[valid_idx].fillna(0)
            y_lead = y_shifted.loc[valid_idx]
            
            # Train models
            trained_models = self.ensemble.train_models(X_lead, y_lead, lead_time_hours)
            
            # Validate
            val_metrics = self._validate_lead_time(X_val, y_val, lead_time_hours)
            
            training_results[lead_time_hours] = {
                'models': trained_models,
                'validation': val_metrics,
                'training_samples': len(X_lead)
            }
            
            # Log results
            tier = self.ensemble.get_tier_from_lead_time(lead_time_hours)
            target_r2 = self.ensemble.performance_targets[tier]
            actual_r2 = val_metrics.get('r2', 0)
            
            status = "✓ PASS" if actual_r2 >= target_r2 else "✗ FAIL"
            logger.info(f"  {lead_time_hours:3d}h ({tier}): R² = {actual_r2:.3f} (target: {target_r2:.3f}) {status}")
            
            performance_summary[lead_time_hours] = {
                'r2': actual_r2,
                'target_r2': target_r2,
                'tier': tier,
                'passed': actual_r2 >= target_r2
            }
        
        # Step 4: Save models
        logger.info("\nStep 4: Saving trained models...")
        self._save_models(training_results)
        
        # Step 5: Performance summary
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("=" * 50)
        
        total_tests = len(performance_summary)
        passed_tests = sum(1 for p in performance_summary.values() if p['passed'])
        
        for lead_time, perf in performance_summary.items():
            days = lead_time / 24
            status = "✓ PASS" if perf['passed'] else "✗ FAIL"
            logger.info(f"{days:4.1f}d ({perf['tier']}): R² = {perf['r2']:.3f} {status}")
        
        overall_grade = self._calculate_grade(passed_tests, total_tests)
        logger.info(f"\nOverall Performance: {overall_grade} ({passed_tests}/{total_tests} passed)")
        
        self.is_trained = True
        self.training_results = training_results
        self.performance_metrics = performance_summary
        
        return {
            'performance_summary': performance_summary,
            'overall_grade': overall_grade,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
    
    def _validate_lead_time(self, X_val: pd.DataFrame, y_val: pd.Series, 
                           lead_time_hours: int) -> Dict:
        """Validate model for specific lead time"""
        
        # Create shifted validation target
        y_val_shifted = y_val.shift(-lead_time_hours // 6)
        valid_idx = y_val_shifted.dropna().index
        
        if len(valid_idx) < 10:
            return {'r2': 0.0, 'rmse': float('inf')}
        
        X_val_lead = X_val.loc[valid_idx].fillna(0)
        y_val_lead = y_val_shifted.loc[valid_idx]
        
        # Make predictions
        predictions = []
        for i in range(len(X_val_lead)):
            try:
                pred_result = self.ensemble.predict(X_val_lead.iloc[i:i+1], lead_time_hours)
                if 'prediction' in pred_result:
                    predictions.append(pred_result['prediction'])
                else:
                    predictions.append(np.nan)
            except:
                predictions.append(np.nan)
        
        # Calculate metrics
        valid_mask = ~np.isnan(predictions)
        if np.sum(valid_mask) < 5:
            return {'r2': 0.0, 'rmse': float('inf')}
        
        y_true = y_val_lead.values[valid_mask]
        y_pred = np.array(predictions)[valid_mask]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'n_samples': len(y_true)}
    
    def _calculate_grade(self, passed: int, total: int) -> str:
        """Calculate overall performance grade"""
        if total == 0:
            return "NO DATA"
        
        pass_rate = passed / total
        if pass_rate >= 1.0:
            return "EXCELLENT"
        elif pass_rate >= 0.8:
            return "GOOD"
        elif pass_rate >= 0.6:
            return "FAIR"
        else:
            return "POOR"
    
    def predict_extended_range(self, lead_times: List[int] = None, 
                              station_id: str = "8575512") -> Dict:
        """Make extended-range predictions"""
        
        if not self.is_trained:
            return {'error': 'System not trained. Run train_system() first.'}
        
        if lead_times is None:
            lead_times = [24, 48, 72, 120, 168, 240, 336]
        
        logger.info(f"Generating extended-range forecast for {len(lead_times)} lead times...")
        
        # Get current data (last 7 days for context)
        historical_data = self.data_manager.load_historical_data(station_id)
        climate_data = self.data_manager.load_climate_data()
        
        if historical_data.empty:
            return {'error': 'No historical data available'}
        
        # Create features for current conditions
        features_df = self.feature_engine.create_features(historical_data, climate_data)
        current_features = features_df.tail(1)  # Most recent observation
        
        # Extract climate state
        climate_state = {}
        for col in current_features.columns:
            if 'nino34_current' in col or 'nao_current' in col:
                climate_state[col] = current_features[col].iloc[0]
        
        # Make predictions for each lead time
        predictions = {}
        current_time = historical_data.index[-1]
        
        for lead_time_hours in lead_times:
            try:
                # Raw prediction
                pred_result = self.ensemble.predict(current_features, lead_time_hours)
                
                if 'prediction' in pred_result:
                    raw_prediction = pred_result['prediction']
                    
                    # Apply corrections
                    corrected_prediction = self.corrector.apply_correction(
                        raw_prediction, current_time, climate_state, lead_time_hours
                    )
                    
                    # Calculate bounds
                    uncertainty = pred_result['uncertainty']
                    lower_bound = corrected_prediction - 2 * uncertainty
                    upper_bound = corrected_prediction + 2 * uncertainty
                    
                    # Forecast time
                    forecast_time = current_time + timedelta(hours=lead_time_hours)
                    
                    predictions[lead_time_hours] = {
                        'prediction': corrected_prediction,
                        'raw_prediction': raw_prediction,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'uncertainty': uncertainty,
                        'confidence': pred_result['confidence'],
                        'tier': pred_result['tier'],
                        'forecast_time': forecast_time.strftime('%Y-%m-%d %H:%M'),
                        'correction_applied': corrected_prediction - raw_prediction
                    }
                else:
                    predictions[lead_time_hours] = {'error': pred_result.get('error', 'Prediction failed')}
                    
            except Exception as e:
                predictions[lead_time_hours] = {'error': str(e)}
        
        # Generate forecast report
        report = self._generate_forecast_report(predictions, current_time)
        
        return {
            'predictions': predictions,
            'forecast_report': report,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_water_level': historical_data['water_level'].iloc[-1],
            'station_id': station_id
        }
    
    def _generate_forecast_report(self, predictions: Dict, current_time: datetime) -> str:
        """Generate human-readable forecast report"""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("EXTENDED-RANGE COASTAL FLOOD FORECAST")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Current Conditions: {current_time.strftime('%Y-%m-%d %H:%M')}")
        report_lines.append("")
        
        # Performance targets
        report_lines.append("SYSTEM PERFORMANCE TARGETS:")
        report_lines.append("• 1-3 days: R² >0.90 (high skill)")
        report_lines.append("• 3-7 days: R² >0.75 (useful skill)")
        report_lines.append("• 7+ days: R² >0.60 (probabilistic guidance)")
        report_lines.append("")
        
        # Forecast summary
        report_lines.append("FORECAST SUMMARY:")
        report_lines.append("-" * 40)
        
        for lead_time, pred_data in sorted(predictions.items()):
            if 'error' in pred_data:
                report_lines.append(f"{lead_time:3d}h: ERROR - {pred_data['error']}")
                continue
            
            days = lead_time / 24
            tier = pred_data['tier']
            prediction = pred_data['prediction']
            uncertainty = pred_data['uncertainty']
            confidence = pred_data['confidence']
            
            report_lines.append(f"{lead_time:3d}h ({days:4.1f}d): {prediction:6.2f} ft "
                               f"±{uncertainty:.2f} (Conf: {confidence:.0%}, {tier.upper()})")
        
        report_lines.append("")
        
        # Flood risk assessment
        report_lines.append("FLOOD RISK ASSESSMENT:")
        report_lines.append("-" * 40)
        
        flood_thresholds = {'minor': 2.6, 'moderate': 3.2, 'major': 4.0}
        
        for lead_time, pred_data in sorted(predictions.items()):
            if 'error' in pred_data:
                continue
            
            days = lead_time / 24
            prediction = pred_data['prediction']
            upper_bound = pred_data['upper_bound']
            
            risk_level = "LOW"
            if prediction > flood_thresholds['minor']:
                risk_level = "MINOR"
            if prediction > flood_thresholds['moderate']:
                risk_level = "MODERATE"
            if prediction > flood_thresholds['major']:
                risk_level = "MAJOR"
            
            # Estimate probability of minor flooding
            prob_minor = min(100, max(0, 100 * (upper_bound - flood_thresholds['minor']) / 
                                    (2 * pred_data['uncertainty'])))
            
            report_lines.append(f"{days:4.1f}d: {risk_level:8s} risk (P(Minor Flood) ≈ {prob_minor:3.0f}%)")
        
        report_lines.append("")
        report_lines.append("NOTES:")
        report_lines.append("• Predictions include climate teleconnection effects")
        report_lines.append("• Uncertainty increases with lead time")
        report_lines.append("• All water levels in feet MLLW")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def _save_models(self, training_results: Dict):
        """Save trained models to disk"""
        
        models_path = self.data_manager.models_dir / "trained_models.pkl"
        
        # Prepare models for saving
        models_to_save = {
            'ensemble': self.ensemble,
            'corrector': self.corrector,
            'training_results': training_results,
            'feature_engine': self.feature_engine,
            'system_version': '1.0.0-standalone'
        }
        
        with open(models_path, 'wb') as f:
            pickle.dump(models_to_save, f)
        
        logger.info(f"Models saved to {models_path}")
    
    def load_models(self):
        """Load previously trained models"""
        
        models_path = self.data_manager.models_dir / "trained_models.pkl"
        
        if not models_path.exists():
            logger.warning("No saved models found")
            return False
        
        try:
            with open(models_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.ensemble = saved_data['ensemble']
            self.corrector = saved_data['corrector']
            self.training_results = saved_data.get('training_results', {})
            
            if 'feature_engine' in saved_data:
                self.feature_engine = saved_data['feature_engine']
            
            self.is_trained = True
            logger.info(f"Models loaded from {models_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

# ====================================================================
# WEB INTERFACE (OPTIONAL)
# ====================================================================

def create_web_interface(system: StandaloneExtendedRangeSystem):
    """Create simple web interface using Flask"""
    
    if not FLASK_AVAILABLE:
        logger.warning("Flask not available - web interface disabled")
        return None
    
    app = Flask(__name__)
    
    # Simple HTML template
    HTML_TEMPLATE = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Extended-Range Coastal Flood Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
            .content { padding: 20px; }
            .forecast { background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .status-good { color: #27ae60; }
            .status-warning { color: #f39c12; }
            .status-danger { color: #e74c3c; }
            pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
            button { background-color: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌊 Extended-Range Coastal Flood Prediction</h1>
            <p>Advanced AI System for 1-14 Day Forecasting</p>
        </div>
        
        <div class="content">
            {% if not system_trained %}
            <div class="forecast">
                <h3>⚠️ System Not Trained</h3>
                <p>The prediction system needs to be trained before generating forecasts.</p>
                <form method="post" action="/train">
                    <button type="submit">Train System</button>
                </form>
            </div>
            {% else %}
            <div class="forecast">
                <h3>✅ System Ready</h3>
                <p>Extended-range prediction system is trained and ready.</p>
                <form method="post" action="/forecast">
                    <button type="submit">Generate Forecast</button>
                </form>
            </div>
            
            {% if forecast_data %}
            <div class="forecast">
                <h3>📊 Latest Forecast</h3>
                <pre>{{ forecast_report }}</pre>
            </div>
            {% endif %}
            
            {% if performance_data %}
            <div class="forecast">
                <h3>📈 System Performance</h3>
                {% for lead_time, perf in performance_data.items() %}
                <p>
                    <strong>{{ (lead_time / 24)|round(1) }} days ({{ perf.tier }}):</strong>
                    <span class="{% if perf.passed %}status-good{% else %}status-danger{% endif %}">
                        R² = {{ "%.3f"|format(perf.r2) }} (target: {{ "%.3f"|format(perf.target_r2) }})
                        {% if perf.passed %}✓{% else %}✗{% endif %}
                    </span>
                </p>
                {% endfor %}
            </div>
            {% endif %}
            {% endif %}
        </div>
    </body>
    </html>
    '''
    
    @app.route('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            system_trained=system.is_trained,
            performance_data=system.performance_metrics,
            forecast_data=None,
            forecast_report=""
        )
    
    @app.route('/train', methods=['POST'])
    def train():
        logger.info("Training system via web interface...")
        results = system.train_system()
        return render_template_string(
            HTML_TEMPLATE,
            system_trained=system.is_trained,
            performance_data=system.performance_metrics,
            forecast_data=None,
            forecast_report=f"Training completed with grade: {results.get('overall_grade', 'Unknown')}"
        )
    
    @app.route('/forecast', methods=['POST'])
    def forecast():
        logger.info("Generating forecast via web interface...")
        results = system.predict_extended_range()
        return render_template_string(
            HTML_TEMPLATE,
            system_trained=system.is_trained,
            performance_data=system.performance_metrics,
            forecast_data=results,
            forecast_report=results.get('forecast_report', 'No forecast available')
        )
    
    @app.route('/api/forecast')
    def api_forecast():
        """JSON API endpoint"""
        if not system.is_trained:
            return jsonify({'error': 'System not trained'})
        
        results = system.predict_extended_range()
        return jsonify(results)
    
    return app

# ====================================================================
# COMMAND LINE INTERFACE
# ====================================================================

def main():
    """Main command line interface"""
    
    print("🌊 Extended-Range Coastal Flood Prediction System")
    print("=" * 50)
    
    # Initialize system
    system = StandaloneExtendedRangeSystem()
    
    # Try to load existing models
    if system.load_models():
        print("✓ Loaded existing trained models")
    else:
        print("ℹ No existing models found")
    
    while True:
        print("\nAvailable commands:")
        print("1. Train system")
        print("2. Generate forecast")
        print("3. View performance")
        print("4. Start web interface")
        print("5. Exit")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                print("\nTraining system...")
                results = system.train_system()
                print(f"\nTraining completed with grade: {results.get('overall_grade', 'Unknown')}")
                
            elif choice == '2':
                if not system.is_trained:
                    print("❌ System not trained. Please train first (option 1).")
                    continue
                
                print("\nGenerating extended-range forecast...")
                results = system.predict_extended_range()
                
                if 'error' in results:
                    print(f"❌ Error: {results['error']}")
                else:
                    print("\n" + results['forecast_report'])
                
            elif choice == '3':
                if not system.performance_metrics:
                    print("❌ No performance data available. Please train first.")
                    continue
                
                print("\nSystem Performance:")
                print("-" * 30)
                for lead_time, perf in system.performance_metrics.items():
                    days = lead_time / 24
                    status = "✓ PASS" if perf['passed'] else "✗ FAIL"
                    print(f"{days:4.1f}d ({perf['tier']}): R² = {perf['r2']:.3f} {status}")
                
            elif choice == '4':
                if FLASK_AVAILABLE:
                    print("\nStarting web interface...")
                    app = create_web_interface(system)
                    if app:
                        print("🌐 Web interface available at: http://localhost:5000")
                        print("Press Ctrl+C to stop")
                        app.run(host='0.0.0.0', port=5000, debug=False)
                else:
                    print("❌ Flask not available. Install with: pip install flask")
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()