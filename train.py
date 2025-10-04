"""
Main training script for ML trading models.
"""
import argparse
from datetime import datetime, timedelta
from src.training.train_pipeline import TrainingPipeline


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML trading model')
    
    parser.add_argument('--symbol', type=str, default='^BVSP',
                        help='Trading symbol (default: ^BVSP)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'logistic'],
                        help='Model type (default: xgboost)')
    parser.add_argument('--target-type', type=str, default='direction',
                        choices=['direction', 'returns', 'binary'],
                        help='Target type (default: direction)')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon in days (default: 1)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save trained model')
    parser.add_argument('--use-mlflow', action='store_true',
                        help='Use MLflow for experiment tracking')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    if args.start_date is None:
        start = datetime.now() - timedelta(days=730)  # 2 years
        args.start_date = start.strftime('%Y-%m-%d')
    
    print("=" * 80)
    print("ML Trading Model Training")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Model Type: {args.model_type}")
    print(f"Target Type: {args.target_type}")
    print(f"Horizon: {args.horizon} day(s)")
    print("=" * 80)
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        model_type=args.model_type,
        target_type=args.target_type,
        horizon=args.horizon
    )
    
    results = pipeline.run(use_mlflow=args.use_mlflow)
    
    # Print results
    print("\n" + "=" * 80)
    print("Training Results")
    print("=" * 80)
    
    print("\nTest Metrics:")
    for key, value in results['test_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTop 10 Most Important Features:")
    print(results['feature_importance'].head(10).to_string(index=False))
    
    # Save model if requested
    if args.save_model:
        pipeline.save_model(args.save_model)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
