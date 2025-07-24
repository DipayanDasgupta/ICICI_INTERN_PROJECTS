# train_and_save_model.py (Corrected to use the local database)

import joblib
import pandas as pd
import lightgbm as lgb
import sqlite3
from flask import Flask
from datetime import date, timedelta
from tqdm import tqdm

# Correctly import the necessary components from your 'app' package
from app.config import DB_FILE  # Import the database filename
from app.data_fetcher import get_stock_universe, get_historical_data
from app.strategy import generate_all_features

def train_production_model(symbols):
    """Trains the final LightGBM model using data from the local database."""
    print("--- Fetching data for production model from local DB... ---")
    end_date = date.today()
    start_date = end_date - timedelta(days=4 * 365)  # 3 years for training + 1 for buffer

    all_training_data = []

    # Fetch benchmark data once for efficiency
    benchmark_df = get_historical_data('^NSEI', start_date, end_date)
    if benchmark_df.empty:
        print("FATAL: Could not fetch benchmark data (^NSEI) from the database. Cannot train model.")
        return None

    print("--- Preparing training data for all symbols... ---")
    for symbol in tqdm(symbols, desc="Processing Symbols from DB"):
        # This function now reads from your market_data.db
        data = get_historical_data(symbol, start_date, end_date)
        if not data.empty and len(data) > 252:
            all_features_df = generate_all_features(data, benchmark_df)
            
            if 'Target' in all_features_df.columns:
                # Drop rows where the target (future return) could not be calculated
                training_ready_df = all_features_df.dropna(subset=['Target'])
                if not training_ready_df.empty:
                    all_training_data.append(training_ready_df)

    if not all_training_data:
        print("Could not generate any training features. Aborting.")
        return None

    full_dataset = pd.concat(all_training_data, ignore_index=True)

    feature_cols = [
        'MA_20', 'MA_50', 'ROC_20', 'Volatility_20D', 'RSI', 'Relative_Strength',
        'Momentum_3M', 'Momentum_6M', 'Momentum_12M', 'Sharpe_3M'
    ]

    # Final cleanup: ensure no NaNs exist in feature or target columns
    full_dataset.dropna(subset=feature_cols + ['Target'], inplace=True)
    X = full_dataset[feature_cols]
    y = full_dataset['Target']

    best_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
        'verbosity': -1, 'boosting_type': 'gbdt', 'n_jobs': -1,
        'lambda_l1': 0.000000035570592833340485, 'lambda_l2': 0.00000023634663148674545, 'num_leaves': 250,
        'feature_fraction': 0.43246049791683877, 'bagging_fraction':0.6098643301693437, 'bagging_freq': 4,
        'min_child_samples': 7,
    }

    print(f"--- Training final LightGBM model on {len(X)} data points... ---")
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X, y)
    print("Model training complete.")
    return model

def run_training_pipeline():
    """Main training workflow."""
    # A standalone script needs to create a minimal Flask app to provide a
    # "context" for functions that are part of the main application.
    app = Flask(__name__)
    
    with app.app_context():
        print("--- Using NIFTY_500 universe for training. ---")
        training_symbols = get_stock_universe("NIFTY_500")
        
        model = None
        if training_symbols:
            # Connect to the DB to filter out symbols that we know have no data
            with sqlite3.connect(DB_FILE) as conn:
                meta_df = pd.read_sql_query("SELECT Symbol, Sector FROM stock_metadata", conn)
            
            valid_symbols = meta_df[meta_df['Sector'] != 'No Data']['Symbol'].tolist()
            
            # Use only the symbols that are both in the NIFTY 500 list and have valid data
            final_training_symbols = [s for s in training_symbols if s in valid_symbols]
            
            print(f"--- Found {len(final_training_symbols)} valid symbols in the universe to train on. ---")
            
            if final_training_symbols:
                model = train_production_model(final_training_symbols)

        if model:
            model_filename = 'app/stock_selector_model.joblib'
            joblib.dump(model, model_filename)
            print(f"\nModel successfully trained on NIFTY 500 and saved to {model_filename}")
        else:
            print("\nModel training failed. Please check the logs.")

if __name__ == '__main__':
    run_training_pipeline()