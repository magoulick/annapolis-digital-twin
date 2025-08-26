# Annapolis Digital Twin - Machine Learning Models
# This script implements various ML models for extreme water level prediction

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import neural network libraries
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras not available. LSTM models will be skipped.")
    KERAS_AVAILABLE = False

class AnnapolisMLPredictor:
    def __init__(self, data_file='processed_annapolis_data.csv'):
        self.data_file = data_file
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'regressor': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                'classifier': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            },
            'xgboost': {
                'regressor': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'classifier': xgb.XGBClassifier(random_state=42, n_jobs=-1)
            },
            'gradient_boosting': {
                'regressor': GradientBoostingRegressor(random_state=42),
                'classifier': None  # Will use XGB for classification
            }
        }
        
        # Evaluation metrics storage
        self.evaluation_results = {}
    
    def load_and_prepare_data(self):
        """Load processed data and prepare for ML"""
        print("Loading processed data...")
        
        try:
            df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
            print(f"Loaded data shape: {df.shape}")
        except FileNotFoundError:
            print(f"Error: {self.data_file} not found. Please run data processing first.")
            return None
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Identify feature columns (exclude labels and raw data)
        exclude_patterns = ['label_', '_ahead', 'station_id', 'site_code', 'quality_flag']
        self.feature_columns = [col for col in df.columns 
                               if not any(pattern in col for pattern in exclude_patterns)]
        
        # Identify target columns
        self.target_columns = [col for col in df.columns if 'label_' in col or '_ahead' in col]
        
        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns[:10]}...")
        print(f"Target columns ({len(self.target_columns)}): {self.target_columns[:10]}...")
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix with proper handling of missing values"""
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        # For time series, forward fill then backward fill
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN, fill with column median
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Remove any remaining NaN rows
        X = X.dropna()
        
        return X
    
    def create_sequences_for_lstm(self, X, y, sequence_length=24):
        """Create sequences for LSTM model (24 = 4 hours of 6-minute data)"""
        if not KERAS_AVAILABLE:
            return None, None
        
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_regression_models(self, df, target_col='water_level_6h_ahead', test_size=0.2):
        """Train regression models for continuous water level prediction"""
        print(f"\nTraining regression models for target: {target_col}")
        
        if target_col not in df.columns:
            print(f"Target column {target_col} not found in data")
            return
        
        # Prepare data
        X = self.prepare_features(df)
        y = df[target_col].reindex(X.index).dropna()
        
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 100:
            print(f"Not enough data points ({len(X)}) for training")
            return
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split data for final evaluation
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[f'{target_col}_scaler'] = scaler
        
        results = {}
        
        # Train traditional ML models
        for model_name, config in self.model_configs.items():
            if config['regressor'] is None:
                continue
                
            print(f"Training {model_name}...")
            model = config['regressor']
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Evaluate
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[model_name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'predictions': y_pred_test,
                'actual': y_test
            }
            
            print(f"  {model_name} - Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
        
        # Train LSTM model if available
        if KERAS_AVAILABLE and len(X_train) > 100:
            print("Training LSTM model...")
            
            # Create sequences
            X_seq_train, y_seq_train = self.create_sequences_for_lstm(
                pd.DataFrame(X_train_scaled), y_train
            )
            X_seq_test, y_seq_test = self.create_sequences_for_lstm(
                pd.DataFrame(X_test_scaled), y_test
            )
            
            if X_seq_train is not None and len(X_seq_train) > 50:
                # Build LSTM model
                lstm_model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(24, X_train_scaled.shape[1])),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1)
                ])
                
                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
                
                # Callbacks
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
                
                # Train
                history = lstm_model.fit(
                    X_seq_train, y_seq_train,
                    validation_data=(X_seq_test, y_seq_test),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                # Evaluate
                y_pred_lstm = lstm_model.predict(X_seq_test).flatten()
                lstm_rmse = np.sqrt(mean_squared_error(y_seq_test, y_pred_lstm))
                lstm_mae = mean_absolute_error(y_seq_test, y_pred_lstm)
                lstm_r2 = r2_score(y_seq_test, y_pred_lstm)
                
                results['lstm'] = {
                    'model': lstm_model,
                    'train_rmse': np.nan,  # Not computed for LSTM
                    'test_rmse': lstm_rmse,
                    'test_mae': lstm_mae,
                    'test_r2': lstm_r2,
                    'predictions': y_pred_lstm,
                    'actual': y_seq_test,
                    'history': history
                }
                
                print(f"  LSTM - Test RMSE: {lstm_rmse:.3f}, R²: {lstm_r2:.3f}")
        
        # Store results
        self.models[f'{target_col}_regression'] = results
        self.evaluation_results[f'{target_col}_regression'] = results
        
        return results
    
    def train_classification_models(self, df, target_col='label_minor_flood_6h_lead', test_size=0.2):
        """Train classification models for flood prediction"""
        print(f"\nTraining classification models for target: {target_col}")
        
        if target_col not in df.columns:
            print(f"Target column {target_col} not found in data")
            return
        
        # Prepare data
        X = self.prepare_features(df)
        y = df[target_col].reindex(X.index).dropna()
        
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 100:
            print(f"Not enough data points ({len(X)}) for training")
            return
        
        # Check class balance
        class_counts = y.value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            print("Only one class present in target variable")
            return
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[f'{target_col}_scaler'] = scaler
        
        results = {}
        
        # Train classification models
        for model_name, config in self.model_configs.items():
            if config['classifier'] is None:
                continue
                
            print(f"Training {model_name} classifier...")
            model = config['classifier']
            
            # Handle class imbalance
            if hasattr(model, 'class_weight'):
                model.set_params(class_weight='balanced')
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, zero_division=0)
            recall = recall_score(y_test, y_pred_test, zero_division=0)
            f1 = f1_score(y_test, y_pred_test, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'predictions': y_pred_test,
                'probabilities': y_pred_proba,
                'actual': y_test
            }
            
            print(f"  {model_name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Store results
        self.models[f'{target_col}_classification'] = results
        self.evaluation_results[f'{target_col}_classification'] = results
        
        return results
    
    def create_evaluation_plots(self, results, model_type='regression'):
        """Create evaluation plots for model results"""
        if not results:
            return
        
        if model_type == 'regression':
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Model comparison
            model_names = list(results.keys())
            test_rmse = [results[name]['test_rmse'] for name in model_names]
            test_r2 = [results[name]['test_r2'] for name in model_names]
            
            axes[0,0].bar(model_names, test_rmse)
            axes[0,0].set_title('Model Comparison - Test RMSE')
            axes[0,0].set_ylabel('RMSE (feet)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            axes[0,1].bar(model_names, test_r2)
            axes[0,1].set_title('Model Comparison - R² Score')
            axes[0,1].set_ylabel('R² Score')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Best model predictions
            best_model = min(results.keys(), key=lambda x: results[x]['test_rmse'])
            best_results = results[best_model]
            
            # Time series plot
            axes[1,0].plot(best_results['actual'].values, label='Actual', alpha=0.7)
            axes[1,0].plot(best_results['predictions'], label='Predicted', alpha=0.7)
            axes[1,0].set_title(f'{best_model} - Time Series Prediction')
            axes[1,0].set_ylabel('Water Level (feet)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Scatter plot
            axes[1,1].scatter(best_results['actual'], best_results['predictions'], alpha=0.6)
            axes[1,1].plot([best_results['actual'].min(), best_results['actual'].max()], 
                          [best_results['actual'].min(), best_results['actual'].max()], 'r--')
            axes[1,1].set_xlabel('Actual Water Level (feet)')
            axes[1,1].set_ylabel('Predicted Water Level (feet)')
            axes[1,1].set_title(f'{best_model} - Actual vs Predicted')
            axes[1,1].grid(True, alpha=0.3)
            
        else:  # classification
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Model metrics comparison
            model_names = list(results.keys())
            f1_scores = [results[name]['f1_score'] for name in model_names]
            auc_scores = [results[name]['auc'] for name in model_names if not np.isnan(results[name]['auc'])]
            
            axes[0,0].bar(model_names, f1_scores)
            axes[0,0].set_title('Model Comparison - F1 Score')
            axes[0,0].set_ylabel('F1 Score')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            if auc_scores:
                axes[0,1].bar(model_names[:len(auc_scores)], auc_scores)
                axes[0,1].set_title('Model Comparison - AUC Score')
                axes[0,1].set_ylabel('AUC Score')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # Confusion matrix for best model
            best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
            best_results = results[best_model]
            
            cm = confusion_matrix(best_results['actual'], best_results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0], cmap='Blues')
            axes[1,0].set_title(f'{best_model} - Confusion Matrix')
            axes[1,0].set_xlabel('Predicted')
            axes[1,0].set_ylabel('Actual')
            
            # ROC curve if probabilities available
            if best_results['probabilities'] is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(best_results['actual'], best_results['probabilities'])
                axes[1,1].plot(fpr, tpr, label=f'{best_model} (AUC = {best_results["auc"]:.3f})')
                axes[1,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[1,1].set_xlabel('False Positive Rate')
                axes[1,1].set_ylabel('True Positive Rate')
                axes[1,1].set_title('ROC Curve')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'annapolis_ml_{model_type}_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, model_prefix='annapolis_model'):
        """Save trained models and scalers"""
        print("Saving models...")
        
        for model_group, results in self.models.items():
            for model_name, model_data in results.items():
                if 'model' in model_data:
                    filename = f"{model_prefix}_{model_group}_{model_name}.joblib"
                    joblib.dump(model_data['model'], filename)
                    print(f"Saved {filename}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            filename = f"{model_prefix}_{scaler_name}.joblib"
            joblib.dump(scaler, filename)
            print(f"Saved {filename}")
    
    def load_models(self, model_prefix='annapolis_model'):
        """Load saved models and scalers"""
        print("Loading models...")
        # Implementation would load the saved models
        # This is a placeholder for the complete implementation
        pass
    
    def predict_extreme_events(self, df, hours_ahead=6):
        """Make predictions for extreme water level events"""
        print(f"Making predictions for {hours_ahead} hours ahead...")
        
        # Prepare features
        X = self.prepare_features(df)
        
        if X.empty:
            print("No data available for prediction")
            return None
        
        # Get latest data point
        latest_data = X.iloc[-1:].values
        
        # Make predictions with all available models
        predictions = {}
        
        for model_group, results in self.models.items():
            if f'{hours_ahead}h' in model_group:
                for model_name, model_data in results.items():
                    if 'model' in model_data:
                        model = model_data['model']
                        
                        # Scale data if scaler available
                        scaler_key = f"{model_group.split('_')[0]}_{hours_ahead}h_ahead_scaler"
                        if scaler_key in self.scalers:
                            latest_scaled = self.scalers[scaler_key].transform(latest_data)
                        else:
                            latest_scaled = latest_data
                        
                        try:
                            if 'classification' in model_group:
                                pred = model.predict(latest_scaled)[0]
                                prob = model.predict_proba(latest_scaled)[0][1] if hasattr(model, 'predict_proba') else None
                                predictions[f"{model_name}_flood_prediction"] = pred
                                if prob is not None:
                                    predictions[f"{model_name}_flood_probability"] = prob
                            else:
                                pred = model.predict(latest_scaled)[0]
                                predictions[f"{model_name}_water_level"] = pred
                        except Exception as e:
                            print(f"Error making prediction with {model_name}: {e}")
        
        return predictions
    
    def generate_feature_importance(self, model_group=None):
        """Generate feature importance plots"""
        if not self.models:
            print("No trained models available")
            return
        
        # Select model group
        if model_group is None:
            model_group = list(self.models.keys())[0]
        
        if model_group not in self.models:
            print(f"Model group {model_group} not found")
            return
        
        results = self.models[model_group]
        
        # Get feature importance from tree-based models
        for model_name, model_data in results.items():
            if 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
                model = model_data['model']
                importances = model.feature_importances_
                
                # Create feature importance plot
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Plot top 20 features
                plt.figure(figsize=(10, 8))
                top_features = feature_importance_df.head(20)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importances - {model_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'annapolis_feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"\nTop 10 most important features for {model_name}:")
                for i, row in top_features.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def run_comprehensive_evaluation(self, df):
        """Run comprehensive model evaluation"""
        print("Starting comprehensive model evaluation...")
        
        # Define targets to evaluate
        regression_targets = [
            'water_level_1h_ahead',
            'water_level_6h_ahead',
            'water_level_12h_ahead'
        ]
        
        classification_targets = [
            'label_minor_flood_1h_lead',
            'label_minor_flood_6h_lead',
            'label_moderate_flood_6h_lead'
        ]
        
        # Train regression models
        for target in regression_targets:
            if target in df.columns:
                results = self.train_regression_models(df, target)
                if results:
                    self.create_evaluation_plots(results, 'regression')
        
        # Train classification models
        for target in classification_targets:
            if target in df.columns:
                results = self.train_classification_models(df, target)
                if results:
                    self.create_evaluation_plots(results, 'classification')
        
        # Generate feature importance
        if self.models:
            self.generate_feature_importance()
        
        # Save models
        self.save_models()
        
        print("Comprehensive evaluation completed!")
    
    def create_model_performance_summary(self):
        """Create a summary of all model performances"""
        if not self.evaluation_results:
            print("No evaluation results available")
            return
        
        print("\n=== MODEL PERFORMANCE SUMMARY ===")
        
        # Regression models summary
        regression_summary = []
        for model_group, results in self.evaluation_results.items():
            if 'regression' in model_group:
                target = model_group.replace('_regression', '')
                for model_name, metrics in results.items():
                    regression_summary.append({
                        'Target': target,
                        'Model': model_name,
                        'RMSE': metrics['test_rmse'],
                        'MAE': metrics['test_mae'],
                        'R²': metrics['test_r2']
                    })
        
        if regression_summary:
            regression_df = pd.DataFrame(regression_summary)
            print("\nRegression Models:")
            print(regression_df.to_string(index=False, float_format='%.3f'))
        
        # Classification models summary
        classification_summary = []
        for model_group, results in self.evaluation_results.items():
            if 'classification' in model_group:
                target = model_group.replace('_classification', '')
                for model_name, metrics in results.items():
                    classification_summary.append({
                        'Target': target,
                        'Model': model_name,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1': metrics['f1_score'],
                        'AUC': metrics['auc']
                    })
        
        if classification_summary:
            classification_df = pd.DataFrame(classification_summary)
            print("\nClassification Models:")
            print(classification_df.to_string(index=False, float_format='%.3f'))
    
    def create_real_time_prediction_system(self, db_path='annapolis_digital_twin.db'):
        """Create a real-time prediction system using the trained models"""
        print("Setting up real-time prediction system...")
        
        def make_real_time_prediction():
            """Function to make real-time predictions"""
            # Load latest data from database
            conn = sqlite3.connect(db_path)
            
            # Get latest 24 hours of data for feature creation
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Load tide data
            tide_query = f"""
                SELECT * FROM tide_data 
                WHERE timestamp >= '{start_time_str}'
                ORDER BY timestamp DESC
                LIMIT 240
            """
            tide_df = pd.read_sql_query(tide_query, conn)
            
            # Load weather data
            weather_query = f"""
                SELECT * FROM weather_data 
                WHERE timestamp >= '{start_time_str}'
                ORDER BY timestamp DESC
                LIMIT 240
            """
            weather_df = pd.read_sql_query(weather_query, conn)
            
            conn.close()
            
            if tide_df.empty:
                print("No recent tide data available")
                return None
            
            # Process data (simplified version)
            from annapolis_data_processing import AnnapolisDataProcessor
            processor = AnnapolisDataProcessor(db_path)
            
            # This would use the full processing pipeline
            # For demonstration, we'll use a simplified approach
            if not tide_df.empty and 'water_level' in tide_df.columns:
                current_level = tide_df['water_level'].iloc[0]
                level_trend = tide_df['water_level'].iloc[0] - tide_df['water_level'].iloc[10] if len(tide_df) > 10 else 0
                
                # Simple prediction based on trend
                predictions = {
                    'current_water_level': current_level,
                    'level_trend_1h': level_trend,
                    'prediction_timestamp': datetime.now().isoformat(),
                    'flood_risk_minor': 'High' if current_level > 2.0 else 'Low',
                    'flood_risk_moderate': 'High' if current_level > 2.5 else 'Low'
                }
                
                return predictions
            
            return None
        
        return make_real_time_prediction

if __name__ == "__main__":
    # Initialize ML predictor
    predictor = AnnapolisMLPredictor()
    
    # Load data
    df = predictor.load_and_prepare_data()
    
    if df is not None:
        # Run comprehensive evaluation
        predictor.run_comprehensive_evaluation(df)
        
        # Create performance summary
        predictor.create_model_performance_summary()
        
        # Demo real-time prediction
        real_time_predictor = predictor.create_real_time_prediction_system()
        latest_prediction = real_time_predictor()
        
        if latest_prediction:
            print("\n=== LATEST REAL-TIME PREDICTION ===")
            for key, value in latest_prediction.items():
                print(f"{key}: {value}")
    
    print("\nML model training and evaluation complete!")