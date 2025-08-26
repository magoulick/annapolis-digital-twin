# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 08:09:24 2025

@author: magoulic
"""

# Annapolis Digital Twin - Data Collection Module
# This script collects real-time data from available sensors around Annapolis, MD

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import sqlite3
import time
import schedule
import warnings
warnings.filterwarnings('ignore')

class AnnapolisDataCollector:
    def __init__(self, db_path='annapolis_digital_twin.db'):
        self.db_path = db_path
        self.setup_database()
        
        # Annapolis-specific station IDs and coordinates
        self.stations = {
            'annapolis_tide': '8575512',  # Annapolis, MD tide gauge
            'baltimore_tide': '8574680',   # Baltimore, MD (backup/comparison)
            'severn_river_usgs': '01647000',  # USGS Severn River near Crownsville
            'spa_creek_coords': (38.9784, -76.4951),  # Spa Creek approximate center
            'annapolis_coords': (38.9784, -76.4951)
        }
        
    def setup_database(self):
        """Initialize SQLite database with tables for different data types"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tide data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tide_data (
                timestamp TEXT PRIMARY KEY,
                station_id TEXT,
                water_level REAL,
                sigma REAL,
                quality_flag TEXT,
                prediction REAL
            )
        ''')
        
        # Weather data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                timestamp TEXT PRIMARY KEY,
                station_id TEXT,
                air_temp REAL,
                water_temp REAL,
                wind_speed REAL,
                wind_direction REAL,
                wind_gust REAL,
                barometric_pressure REAL
            )
        ''')
        
        # USGS stream data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stream_data (
                timestamp TEXT PRIMARY KEY,
                site_code TEXT,
                discharge REAL,
                gage_height REAL,
                temperature REAL
            )
        ''')
        
        # Precipitation data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precipitation_data (
                timestamp TEXT PRIMARY KEY,
                station_id TEXT,
                precipitation REAL,
                location_lat REAL,
                location_lon REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    
    def collect_noaa_tide_data(self, hours_back=24, station_id='8575512'):
        """Collect tide data from NOAA CO-OPS API for Annapolis"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Format dates for NOAA API
        begin_date = start_time.strftime('%Y%m%d %H:%M')
        end_date = end_time.strftime('%Y%m%d %H:%M')
        
        # Water level data
        url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        params = {
            'begin_date': begin_date,
            'end_date': end_date,
            'station': station_id,
            'product': 'water_level',
            'datum': 'MLLW',
            'time_zone': 'lst_ldt',
            'units': 'english',
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    df['timestamp'] = pd.to_datetime(df['t'])
                    df['water_level'] = pd.to_numeric(df['v'], errors='coerce')
                    df['quality_flag'] = df['f']
                    df['station_id'] = station_id
                    
                    # Get predictions for comparison
                    pred_params = params.copy()
                    pred_params['product'] = 'predictions'
                    pred_response = requests.get(url, params=pred_params, timeout=30)
                    
                    if pred_response.status_code == 200:
                        pred_data = pred_response.json()
                        if 'predictions' in pred_data:
                            pred_df = pd.DataFrame(pred_data['predictions'])
                            pred_df['timestamp'] = pd.to_datetime(pred_df['t'])
                            pred_df['prediction'] = pd.to_numeric(pred_df['v'], errors='coerce')
                            
                            # Merge predictions with observations
                            df = df.merge(pred_df[['timestamp', 'prediction']], 
                                        on='timestamp', how='left')
                    
                    return df[['timestamp', 'station_id', 'water_level', 'quality_flag', 'prediction']]
                    
        except Exception as e:
            print(f"Error collecting tide data: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame()
    
    def collect_noaa_weather_data(self, hours_back=24, station_id='8575512'):
        """Collect meteorological data from NOAA station"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        begin_date = start_time.strftime('%Y%m%d %H:%M')
        end_date = end_time.strftime('%Y%m%d %H:%M')
        
        weather_products = ['air_temperature', 'water_temperature', 'wind', 'air_pressure']
        weather_data = {}
        
        for product in weather_products:
            url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
            params = {
                'begin_date': begin_date,
                'end_date': end_date,
                'station': station_id,
                'product': product,
                'time_zone': 'lst_ldt',
                'units': 'english',
                'format': 'json'
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                        df['timestamp'] = pd.to_datetime(df['t'])
                        
                        if product == 'wind':
                            df['wind_speed'] = pd.to_numeric(df['s'], errors='coerce')
                            df['wind_direction'] = pd.to_numeric(df['d'], errors='coerce')
                            df['wind_gust'] = pd.to_numeric(df['g'], errors='coerce')
                            weather_data['wind'] = df[['timestamp', 'wind_speed', 'wind_direction', 'wind_gust']]
                        else:
                            df['value'] = pd.to_numeric(df['v'], errors='coerce')
                            weather_data[product] = df[['timestamp', 'value']]
                            
            except Exception as e:
                print(f"Error collecting {product} data: {e}")
        
        # Combine all weather data
        if weather_data:
            combined_df = None
            for product, df in weather_data.items():
                if combined_df is None:
                    combined_df = df
                    if product != 'wind':
                        combined_df = combined_df.rename(columns={'value': product})
                else:
                    combined_df = combined_df.merge(df, on='timestamp', how='outer')
                    if product != 'wind':
                        combined_df = combined_df.rename(columns={'value': product})
            
            combined_df['station_id'] = station_id
            return combined_df
        
        return pd.DataFrame()
    
    def collect_usgs_stream_data(self, site_code='01647000', hours_back=24):
        """Collect USGS stream data for Severn River"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # USGS instantaneous values service
        url = "https://waterservices.usgs.gov/nwis/iv/"
        params = {
            'format': 'json',
            'sites': site_code,
            'startDT': start_time.strftime('%Y-%m-%d'),
            'endDT': end_time.strftime('%Y-%m-%d'),
            'parameterCd': '00060,00065,00010',  # discharge, gage height, temperature
            'siteStatus': 'all'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                if 'value' in data and 'timeSeries' in data['value']:
                    all_data = []
                    
                    for series in data['value']['timeSeries']:
                        parameter = series['variable']['variableCode'][0]['value']
                        parameter_name = series['variable']['variableName']
                        
                        for point in series['values'][0]['value']:
                            timestamp = pd.to_datetime(point['dateTime'])
                            value = float(point['value']) if point['value'] != '' else np.nan
                            
                            all_data.append({
                                'timestamp': timestamp,
                                'parameter': parameter,
                                'parameter_name': parameter_name,
                                'value': value
                            })
                    
                    if all_data:
                        df = pd.DataFrame(all_data)
                        # Pivot to get parameters as columns
                        pivot_df = df.pivot_table(index='timestamp', columns='parameter', values='value')
                        pivot_df = pivot_df.reset_index()
                        
                        # Rename columns based on parameter codes
                        column_mapping = {
                            '00060': 'discharge',  # cubic feet per second
                            '00065': 'gage_height',  # feet
                            '00010': 'temperature'  # celsius
                        }
                        
                        pivot_df = pivot_df.rename(columns=column_mapping)
                        pivot_df['site_code'] = site_code
                        
                        return pivot_df
                        
        except Exception as e:
            print(f"Error collecting USGS data: {e}")
        
        return pd.DataFrame()
    
    def collect_nws_precipitation_data(self, lat=38.9784, lon=-76.4951):
        """Collect precipitation data from National Weather Service"""
        # Get nearest weather stations
        url = f"https://api.weather.gov/points/{lat},{lon}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                # Get forecast office and grid coordinates
                office = data['properties']['cwa']
                grid_x = data['properties']['gridX']
                grid_y = data['properties']['gridY']
                
                # Get current conditions
                forecast_url = f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}"
                forecast_response = requests.get(forecast_url, timeout=30)
                
                if forecast_response.status_code == 200:
                    forecast_data = forecast_response.json()
                    
                    # Extract precipitation data if available
                    if 'properties' in forecast_data:
                        precip_data = []
                        current_time = datetime.now()
                        
                        # This is a simplified example - NWS API structure can be complex
                        precip_data.append({
                            'timestamp': current_time,
                            'station_id': f"{office}_{grid_x}_{grid_y}",
                            'precipitation': 0.0,  # Would need to parse actual precipitation
                            'location_lat': lat,
                            'location_lon': lon
                        })
                        
                        return pd.DataFrame(precip_data)
                        
        except Exception as e:
            print(f"Error collecting NWS data: {e}")
        
        return pd.DataFrame()
    
    def save_to_database(self, df, table_name):
        """Save dataframe to SQLite database"""
        if not df.empty:
            conn = sqlite3.connect(self.db_path)
            
            # Convert timestamp to string for SQLite
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Use replace to handle duplicates
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            print(f"Saved {len(df)} records to {table_name}")
        else:
            print(f"No data to save for {table_name}")
    
    def collect_all_data(self):
        """Main method to collect all available data"""
        print(f"Starting data collection at {datetime.now()}")
        
        # Collect tide data for Annapolis
        print("Collecting tide data...")
        tide_data = self.collect_noaa_tide_data(station_id=self.stations['annapolis_tide'])
        self.save_to_database(tide_data, 'tide_data')
        
        # Collect weather data
        print("Collecting weather data...")
        weather_data = self.collect_noaa_weather_data(station_id=self.stations['annapolis_tide'])
        self.save_to_database(weather_data, 'weather_data')
        
        # Collect USGS stream data for Severn River
        print("Collecting stream data...")
        stream_data = self.collect_usgs_stream_data(site_code=self.stations['severn_river_usgs'])
        self.save_to_database(stream_data, 'stream_data')
        
        # Collect precipitation data
        print("Collecting precipitation data...")
        precip_coords = self.stations['spa_creek_coords']
        precip_data = self.collect_nws_precipitation_data(lat=precip_coords[0], lon=precip_coords[1])
        self.save_to_database(precip_data, 'precipitation_data')
        
        print(f"Data collection completed at {datetime.now()}")
    
    def schedule_data_collection(self):
        """Schedule automatic data collection"""
        # Collect data every 6 minutes (matching NOAA update frequency)
        schedule.every(6).minutes.do(self.collect_all_data)
        
        print("Data collection scheduled every 6 minutes")
        print("Press Ctrl+C to stop...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    # Initialize data collector
    collector = AnnapolisDataCollector()
    
    # Run initial data collection
    collector.collect_all_data()
    
    # Optional: Start scheduled collection (comment out if you just want one-time collection)
    # collector.schedule_data_collection()