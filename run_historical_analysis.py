# Complete Historical Analysis Workflow for Annapolis Digital Twin
# This script runs the complete historical data collection and analysis

import sys
import os
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Annapolis Historical Analysis Workflow')
    parser.add_argument('--years', type=int, default=5, help='Number of years of historical data (default: 5)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD), overrides --years')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection (use existing database)')
    parser.add_argument('--skip-processing', action='store_true', help='Skip data processing')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML model training')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.start_date:
        start_date = args.start_date
        end_date = '2024-12-31'
    else:
        # Default: last N years
        end_date = '2024-12-31'
        start_year = 2024 - args.years + 1
        start_date = f'{start_year}-01-01'
    
    print(f"=== ANNAPOLIS HISTORICAL ANALYSIS WORKFLOW ===")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Steps: Collection={not args.skip_collection}, Processing={not args.skip_processing}, ML={not args.skip_ml}")
    
    # Step 1: Collect Historical Data
    if not args.skip_collection:
        print(f"\n=== STEP 1: COLLECTING HISTORICAL DATA ===")
        try:
            from annapolis_historical_data import AnnapolisHistoricalDataCollector
            
            collector = AnnapolisHistoricalDataCollector()
            results = collector.collect_historical_data_range(start_date, end_date)
            
            print(f"Data collection completed:")
            print(f"- Tide records: {results['tide_records']:,}")
            print(f"- Weather records: {results['weather_records']:,}")
            print(f"- Stream records: {results['stream_records']:,}")
            
        except Exception as e:
            print(f"Error in data collection: {e}")
            print("You may need to install additional dependencies or check API access")
            return False
    else:
        print("Skipping data collection (using existing database)")
    
    # Step 2: Process Historical Data
    if not args.skip_processing:
        print(f"\n=== STEP 2: PROCESSING HISTORICAL DATA ===")
        try:
            from annapolis_historical_processing import AnnapolisHistoricalProcessor
            
            processor = AnnapolisHistoricalProcessor()
            processed_df = processor.process_historical_data(
                start_date=start_date,
                end_date=end_date,
                create_features=True,
                analyze_events=True
            )
            
            if processed_df is not None:
                # Generate comprehensive report
                report_prefix = f'annapolis_historical_{start_date}_{end_date}'
                processor.generate_comprehensive_report(processed_df, report_prefix)
                
                print(f"Data processing completed:")
                print(f"- Final dataset: {processed_df.shape[0]:,} records, {processed_df.shape[1]} features")
                print(f"- Processed data saved to: processed_historical_data_{start_date}_{end_date}.csv")
            else:
                print("Data processing failed - no output generated")
                return False
                
        except Exception as e:
            print(f"Error in data processing: {e}")
            return False
    else:
        print("Skipping data processing")
    
    # Step 3: Train ML Models
    if not args.skip_ml:
        print(f"\n=== STEP 3: TRAINING MACHINE LEARNING MODELS ===")
        try:
            from annapolis_ml_models import AnnapolisMLPredictor
            
            # Use the processed historical data
            data_file = f'processed_historical_data_{start_date}_{end_date}.csv'
            
            if not os.path.exists(data_file):
                print(f"Processed data file not found: {data_file}")
                print("Please run data processing first")
                return False
            
            predictor = AnnapolisMLPredictor(data_file)
            
            # Load and prepare data
            df = predictor.load_and_prepare_data()
            
            if df is not None and len(df) > 1000:  # Need substantial data for training
                print(f"Training models on {len(df):,} records with {len(predictor.feature_columns)} features")
                
                # Run comprehensive evaluation
                predictor.run_comprehensive_evaluation(df)
                
                # Create performance summary
                predictor.create_model_performance_summary()
                
                print("Machine learning model training completed!")
                print("Models saved with prefix: annapolis_model_*")
                
            else:
                print(f"Insufficient data for ML training: {len(df) if df is not None else 0} records")
                print("Need at least 1000 records for robust model training")
                return False
                
        except Exception as e:
            print(f"Error in ML model training: {e}")
            return False
    else:
        print("Skipping ML model training")
    
    print(f"\n=== WORKFLOW COMPLETED SUCCESSFULLY ===")
    print(f"Generated files:")
    print(f"- Historical database: annapolis_digital_twin_historical.db")
    print(f"- Processed data: processed_historical_data_{start_date}_{end_date}.csv")
    print(f"- Extreme events: annapolis_historical_{start_date}_{end_date}_extreme_events.csv")
    print(f"- Analysis plots: annapolis_historical_{start_date}_{end_date}_*.png")
    print(f"- ML models: annapolis_model_*.joblib")
    
    return True

def quick_analysis():
    """Quick analysis function for immediate results"""
    print("=== QUICK HISTORICAL ANALYSIS (Last 2 Years) ===")
    
    # Just run 2 years for faster results
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    
    try:
        # Step 1: Collect data
        print("Collecting 2 years of historical data...")
        from annapolis_historical_data import AnnapolisHistoricalDataCollector
        
        collector = AnnapolisHistoricalDataCollector()
        results = collector.collect_historical_data_range(start_date, end_date)
        
        # Step 2: Process data
        print("Processing data...")
        from annapolis_historical_processing import AnnapolisHistoricalProcessor
        
        processor = AnnapolisHistoricalProcessor()
        processed_df = processor.process_historical_data(
            start_date=start_date,
            end_date=end_date,
            create_features=True,
            analyze_events=True
        )
        
        if processed_df is not None and len(processed_df) > 100:
            # Step 3: Train models
            print("Training ML models...")
            from annapolis_ml_models import AnnapolisMLPredictor
            
            # Save the processed data first
            data_file = f'processed_historical_data_{start_date}_{end_date}.csv'
            processed_df.to_csv(data_file)
            
            predictor = AnnapolisMLPredictor(data_file)
            df = predictor.load_and_prepare_data()
            
            if df is not None:
                # Train key models only for quick analysis
                key_targets = {
                    'regression': ['water_level_6h_ahead'],
                    'classification': ['label_minor_flood_6h_lead']
                }
                
                for target in key_targets['regression']:
                    if target in df.columns:
                        results = predictor.train_regression_models(df, target)
                        if results:
                            predictor.create_evaluation_plots(results, 'regression')
                
                for target in key_targets['classification']:
                    if target in df.columns:
                        results = predictor.train_classification_models(df, target)
                        if results:
                            predictor.create_evaluation_plots(results, 'classification')
                
                predictor.save_models()
                print(f"\nQuick analysis completed!")
                print(f"Dataset: {len(processed_df):,} records, {processed_df.shape[1]} features")
                return True
        
        return False
        
    except Exception as e:
        print(f"Error in quick analysis: {e}")
        return False

def research_ready_analysis():
    """Comprehensive analysis optimized for research paper"""
    print("=== RESEARCH-READY HISTORICAL ANALYSIS ===")
    
    # Collect comprehensive dataset for research
    configs = [
        {
            'name': 'Comprehensive Dataset (2019-2024)',
            'start': '2019-01-01',
            'end': '2024-12-31',
            'description': '5+ years including major storm events'
        },
        {
            'name': 'Storm Events Focus (2017-2022)',
            'start': '2017-06-01', 
            'end': '2022-11-30',
            'description': 'Multiple hurricane seasons for extreme event analysis'
        },
        {
            'name': 'Recent High-Quality (2020-2024)',
            'start': '2020-01-01',
            'end': '2024-12-31', 
            'description': 'Most recent 4+ years with best data quality'
        }
    ]
    
    print("Available research configurations:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}: {config['start']} to {config['end']}")
        print(f"   {config['description']}")
    
    choice = input("\nSelect configuration (1-3, or press Enter for default): ").strip()
    
    if choice == '2':
        selected = configs[1]
    elif choice == '3':
        selected = configs[2]
    else:
        selected = configs[0]  # Default
    
    print(f"\nSelected: {selected['name']}")
    print(f"Date range: {selected['start']} to {selected['end']}")
    
    try:
        # Full workflow with research optimizations
        from annapolis_historical_data import AnnapolisHistoricalDataCollector
        from annapolis_historical_processing import AnnapolisHistoricalProcessor
        from annapolis_ml_models import AnnapolisMLPredictor
        
        # Data collection
        print(f"\nCollecting historical data...")
        collector = AnnapolisHistoricalDataCollector()
        results = collector.collect_historical_data_range(selected['start'], selected['end'])
        
        # Data processing with comprehensive features
        print(f"Processing data with research-grade features...")
        processor = AnnapolisHistoricalProcessor()
        processed_df = processor.process_historical_data(
            start_date=selected['start'],
            end_date=selected['end'],
            create_features=True,
            analyze_events=True
        )
        
        if processed_df is not None:
            # Generate research report
            report_prefix = f'research_analysis_{selected["start"]}_{selected["end"]}'
            processor.generate_comprehensive_report(processed_df, report_prefix)
            
            # ML model training with comprehensive evaluation
            data_file = f'processed_historical_data_{selected["start"]}_{selected["end"]}.csv'
            processed_df.to_csv(data_file)
            
            predictor = AnnapolisMLPredictor(data_file)
            df = predictor.load_and_prepare_data()
            
            if df is not None:
                print(f"Training comprehensive ML models...")
                predictor.run_comprehensive_evaluation(df)
                predictor.create_model_performance_summary()
                
                # Generate feature importance analysis
                predictor.generate_feature_importance()
                
                print(f"\n=== RESEARCH ANALYSIS COMPLETED ===")
                print(f"Dataset: {len(processed_df):,} records, {processed_df.shape[1]} features")
                print(f"\nGenerated research outputs:")
                print(f"- Comprehensive dataset: {data_file}")
                print(f"- Extreme events catalog: {report_prefix}_extreme_events.csv")
                print(f"- Statistical analysis: {report_prefix}_summary_stats.json")
                print(f"- Visualization suite: {report_prefix}_*.png")
                print(f"- ML model suite: annapolis_model_*.joblib")
                print(f"- Feature importance: annapolis_feature_importance_*.png")
                
                # Research metrics summary
                if 'water_level' in processed_df.columns:
                    flood_events = (processed_df['water_level'] > 2.6).sum()
                    data_completeness = (1 - processed_df['water_level'].isna().sum() / len(processed_df)) * 100
                    
                    print(f"\nKey Research Metrics:")
                    print(f"- Data completeness: {data_completeness:.1f}%")
                    print(f"- Minor flood events: {flood_events:,} ({flood_events/len(processed_df)*100:.2f}%)")
                    print(f"- Water level range: {processed_df['water_level'].min():.2f} to {processed_df['water_level'].max():.2f} ft")
                    print(f"- Time span: {(processed_df.index.max() - processed_df.index.min()).days} days")
                
                return True
        
        return False
        
    except Exception as e:
        print(f"Error in research analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode
        print("=== ANNAPOLIS DIGITAL TWIN HISTORICAL ANALYSIS ===")
        print("1. Quick Analysis (2 years, ~30 minutes)")
        print("2. Research-Ready Analysis (5+ years, comprehensive)")
        print("3. Custom Analysis (specify parameters)")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            success = quick_analysis()
        elif choice == '2':
            success = research_ready_analysis()
        else:
            # Run with command line args
            success = main()
        
        if success:
            print("\n🎉 Analysis completed successfully!")
            print("Your historical dataset is ready for research!")
        else:
            print("\n❌ Analysis failed. Check error messages above.")
            sys.exit(1)
    else:
        # Command line mode
        success = main()
        sys.exit(0 if success else 1)