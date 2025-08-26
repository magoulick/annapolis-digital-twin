# Annapolis Digital Twin - Main Pipeline Script
# This script runs the complete pipeline from data collection to predictions

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import schedule
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('annapolis_digital_twin.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our custom modules (assuming they're in the same directory)
try:
    from annapolis_data_collection import AnnapolisDataCollector
    from annapolis_data_processing import AnnapolisDataProcessor
    from annapolis_ml_models import AnnapolisMLPredictor
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure all required modules are in the same directory")
    sys.exit(1)

class AnnapolisDigitalTwin:
    def __init__(self, db_path='annapolis_digital_twin.db', 
                 processed_data_file='processed_annapolis_data.csv'):
        self.db_path = db_path
        self.processed_data_file = processed_data_file
        
        # Initialize components
        self.data_collector = AnnapolisDataCollector(db_path)
        self.data_processor = AnnapolisDataProcessor(db_path)
        self.ml_predictor = AnnapolisMLPredictor(processed_data_file)
        
        # Status tracking
        self.last_data_collection = None
        self.last_model_training = None
        self.models_trained = False
        
        logger.info("Annapolis Digital Twin initialized")
    
    def collect_data(self, force=False):
        """Collect new data from all sources"""
        try:
            current_time = datetime.now()
            
            # Check if we need to collect data (every 6 minutes)
            if not force and self.last_data_collection:
                time_diff = (current_time - self.last_data_collection).total_seconds()
                if time_diff < 360:  # 6 minutes
                    logger.info("Data collection skipped - too recent")
                    return True
            
            logger.info("Starting data collection...")
            self.data_collector.collect_all_data()
            self.last_data_collection = current_time
            
            # Check data quality
            data_quality = self.check_data_quality()
            logger.info(f"Data quality check: {data_quality}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            return False
    
    def process_data(self, hours_back=168):
        """Process collected data and create features"""
        try:
            logger.info("Starting data processing...")
            
            # Process data
            processed_data = self.data_processor.process_all_data(hours_back=hours_back)
            
            if processed_data.empty:
                logger.warning("No processed data generated")
                return False
            
            # Save processed data
            self.data_processor.save_processed_data(processed_data, self.processed_data_file)
            
            # Generate summary
            self.data_processor.generate_data_summary(processed_data)
            
            logger.info(f"Data processing completed. Generated {len(processed_data)} records with {len(processed_data.columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            return False
    
    def train_models(self, retrain_hours=24):
        """Train or retrain ML models"""
        try:
            current_time = datetime.now()
            
            # Check if we need to retrain (every 24 hours or if never trained)
            if self.models_trained and self.last_model_training:
                time_diff = (current_time - self.last_model_training).total_seconds()
                if time_diff < retrain_hours * 3600:
                    logger.info("Model training skipped - models are recent")
                    return True
            
            logger.info("Starting model training...")
            
            # Load processed data
            df = self.ml_predictor.load_and_prepare_data()
            
            if df is None or len(df) < 100:
                logger.warning("Insufficient data for model training")
                return False
            
            # Train models for key targets
            key_targets = {
                'regression': ['water_level_6h_ahead', 'water_level_12h_ahead'],
                'classification': ['label_minor_flood_6h_lead', 'label_moderate_flood_6h_lead']
            }
            
            success_count = 0
            total_targets = sum(len(targets) for targets in key_targets.values())
            
            # Train regression models
            for target in key_targets['regression']:
                if target in df.columns:
                    results = self.ml_predictor.train_regression_models(df, target)
                    if results:
                        success_count += 1
                        logger.info(f"Successfully trained regression models for {target}")
                    else:
                        logger.warning(f"Failed to train regression models for {target}")
            
            # Train classification models
            for target in key_targets['classification']:
                if target in df.columns:
                    results = self.ml_predictor.train_classification_models(df, target)
                    if results:
                        success_count += 1
                        logger.info(f"Successfully trained classification models for {target}")
                    else:
                        logger.warning(f"Failed to train classification models for {target}")
            
            # Save models
            if success_count > 0:
                self.ml_predictor.save_models()
                self.models_trained = True
                self.last_model_training = current_time
                logger.info(f"Model training completed. {success_count}/{total_targets} targets successful")
                return True
            else:
                logger.error("No models were successfully trained")
                return False
                
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False
    
    def make_predictions(self):
        """Make predictions using trained models"""
        try:
            if not self.models_trained:
                logger.warning("No trained models available for predictions")
                return None
            
            logger.info("Making predictions...")
            
            # Load recent processed data
            df = self.ml_predictor.load_and_prepare_data()
            
            if df is None or df.empty:
                logger.warning("No data available for predictions")
                return None
            
            # Make predictions for different time horizons
            predictions = {}
            
            for hours_ahead in [1, 6, 12]:
                pred = self.ml_predictor.predict_extreme_events(df, hours_ahead=hours_ahead)
                if pred:
                    predictions[f'{hours_ahead}h_ahead'] = pred
            
            # Add metadata
            predictions['prediction_time'] = datetime.now().isoformat()
            predictions['data_timestamp'] = df.index[-1].isoformat() if not df.empty else None
            
            # Current conditions
            if not df.empty and 'water_level' in df.columns:
                current_level = df['water_level'].iloc[-1]
                predictions['current_conditions'] = {
                    'water_level': current_level,
                    'minor_flood_status': current_level > 2.6,
                    'moderate_flood_status': current_level > 3.2,
                    'major_flood_status': current_level > 4.2
                }
            
            logger.info(f"Predictions generated for {len(predictions)} scenarios")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def check_data_quality(self):
        """Check the quality of collected data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check recent data availability
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Check tide data
            tide_query = f"SELECT COUNT(*) FROM tide_data WHERE timestamp >= '{start_time_str}'"
            tide_count = pd.read_sql_query(tide_query, conn).iloc[0, 0]
            
            # Check weather data
            weather_query = f"SELECT COUNT(*) FROM weather_data WHERE timestamp >= '{start_time_str}'"
            weather_count = pd.read_sql_query(weather_query, conn).iloc[0, 0]
            
            # Check stream data
            stream_query = f"SELECT COUNT(*) FROM stream_data WHERE timestamp >= '{start_time_str}'"
            stream_count = pd.read_sql_query(stream_query, conn).iloc[0, 0]
            
            conn.close()
            
            quality_score = {
                'tide_data_points': tide_count,
                'weather_data_points': weather_count,
                'stream_data_points': stream_count,
                'overall_quality': 'Good' if tide_count > 5 else 'Poor'
            }
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {'overall_quality': 'Unknown'}
    
    def generate_status_report(self):
        """Generate a comprehensive status report"""
        try:
            logger.info("Generating status report...")
            
            status = {
                'system_status': 'Running',
                'timestamp': datetime.now().isoformat(),
                'last_data_collection': self.last_data_collection.isoformat() if self.last_data_collection else None,
                'last_model_training': self.last_model_training.isoformat() if self.last_model_training else None,
                'models_trained': self.models_trained,
                'data_quality': self.check_data_quality()
            }
            
            # Get latest predictions
            predictions = self.make_predictions()
            if predictions:
                status['latest_predictions'] = predictions
            
            # Database statistics
            conn = sqlite3.connect(self.db_path)
            
            # Total records in each table
            tables = ['tide_data', 'weather_data', 'stream_data', 'precipitation_data']
            record_counts = {}
            
            for table in tables:
                try:
                    count_query = f"SELECT COUNT(*) FROM {table}"
                    count = pd.read_sql_query(count_query, conn).iloc[0, 0]
                    record_counts[table] = count
                except:
                    record_counts[table] = 0
            
            conn.close()
            status['database_records'] = record_counts
            
            # Save status report
            import json
            with open('annapolis_status_report.json', 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
            logger.info("Status report generated")
            return status
            
        except Exception as e:
            logger.error(f"Error generating status report: {e}")
            return None
    
    def run_full_pipeline(self, initial_training=True):
        """Run the complete pipeline once"""
        logger.info("Starting full pipeline execution...")
        
        # Step 1: Collect data
        if not self.collect_data():
            logger.error("Data collection failed")
            return False
        
        # Step 2: Process data
        if not self.process_data():
            logger.error("Data processing failed")
            return False
        
        # Step 3: Train models (if needed)
        if initial_training or not self.models_trained:
            if not self.train_models():
                logger.warning("Model training failed, but continuing...")
        
        # Step 4: Make predictions
        predictions = self.make_predictions()
        if predictions:
            logger.info("Predictions generated successfully")
        else:
            logger.warning("Prediction generation failed")
        
        # Step 5: Generate status report
        status = self.generate_status_report()
        
        logger.info("Full pipeline execution completed")
        return True
    
    def start_continuous_mode(self):
        """Start continuous operation mode"""
        logger.info("Starting continuous operation mode...")
        
        # Schedule data collection every 6 minutes
        schedule.every(6).minutes.do(self.collect_data)
        
        # Schedule data processing every hour
        schedule.every().hour.do(self.process_data)
        
        # Schedule model retraining every 24 hours
        schedule.every(24).hours.do(self.train_models)
        
        # Schedule status reports every 4 hours
        schedule.every(4).hours.do(self.generate_status_report)
        
        # Initial full pipeline run
        self.run_full_pipeline(initial_training=True)
        
        logger.info("Continuous mode started. Press Ctrl+C to stop...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Continuous mode stopped by user")
    
    def emergency_prediction_mode(self):
        """Emergency mode for high-frequency predictions during extreme events"""
        logger.info("Starting emergency prediction mode...")
        
        try:
            while True:
                # Collect data
                self.collect_data(force=True)
                
                # Quick processing of recent data
                self.process_data(hours_back=24)  # Only last 24 hours
                
                # Make predictions
                predictions = self.make_predictions()
                
                if predictions and 'current_conditions' in predictions:
                    current_level = predictions['current_conditions']['water_level']
                    
                    # Check for emergency conditions
                    if current_level > 3.2:  # Moderate flooding
                        logger.critical(f"EMERGENCY: Moderate flooding detected! Water level: {current_level:.2f} ft")
                    elif current_level > 2.6:  # Minor flooding
                        logger.warning(f"WARNING: Minor flooding detected! Water level: {current_level:.2f} ft")
                    
                    # Save emergency predictions
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    emergency_file = f'emergency_predictions_{timestamp}.json'
                    
                    import json
                    with open(emergency_file, 'w') as f:
                        json.dump(predictions, f, indent=2, default=str)
                    
                    logger.info(f"Emergency predictions saved to {emergency_file}")
                
                # Wait 2 minutes before next cycle
                time.sleep(120)
                
        except KeyboardInterrupt:
            logger.info("Emergency mode stopped by user")

def main():
    parser = argparse.ArgumentParser(description='Annapolis Digital Twin System')
    parser.add_argument('--mode', choices=['single', 'continuous', 'emergency'], 
                       default='single', help='Operation mode')
    parser.add_argument('--db-path', default='annapolis_digital_twin.db', 
                       help='Database path')
    parser.add_argument('--data-file', default='processed_annapolis_data.csv', 
                       help='Processed data file path')
    parser.add_argument('--hours-back', type=int, default=168, 
                       help='Hours of historical data to process')
    
    args = parser.parse_args()
    
    # Initialize digital twin
    twin = AnnapolisDigitalTwin(db_path=args.db_path, 
                               processed_data_file=args.data_file)
    
    if args.mode == 'single':
        logger.info("Running single execution mode")
        success = twin.run_full_pipeline()
        if success:
            logger.info("Single execution completed successfully")
        else:
            logger.error("Single execution failed")
            sys.exit(1)
    
    elif args.mode == 'continuous':
        logger.info("Running continuous mode")
        twin.start_continuous_mode()
    
    elif args.mode == 'emergency':
        logger.info("Running emergency prediction mode")
        twin.emergency_prediction_mode()

if __name__ == "__main__":
    main()