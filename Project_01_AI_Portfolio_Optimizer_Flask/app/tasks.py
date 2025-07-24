# app/tasks.py (Corrected)

from celery import shared_task
from .backtesting import run_backtest, run_custom_portfolio_backtest
import traceback
import sqlite3
import json
import joblib
from .config import PORTFOLIOS_DB_FILE
from . import ml_models
from . import data_fetcher
# --- THIS IS THE CORRECTED FUNCTION ---
@shared_task(bind=True)
def run_backtest_task(self, start_date_str, end_date_str, universe_name, top_n, risk_free_rate, optimization_method, sector_constraints=None):
    """Celery task for the ML-driven strategy."""
    def progress_callback(message):
        self.update_state(state='PROGRESS', meta={'status': message})
    try:
        # The function in backtesting.py now receives the new argument
        results = run_backtest(
            start_date_str=start_date_str, 
            end_date_str=end_date_str, 
            universe_name=universe_name, 
            top_n=top_n, 
            risk_free_rate=risk_free_rate, 
            progress_callback=progress_callback,
            optimization_method=optimization_method, # Pass it down
            sector_constraints=sector_constraints
        )
        return results
    except Exception as e:
        traceback.print_exc()
        # Re-raising the exception will mark the task as FAILED in Celery
        raise e

@shared_task(bind=True)
def run_custom_backtest_task(self, portfolio_id, start_date_str, end_date_str, risk_free_rate, universe_name):
    """Celery task for custom user-defined portfolios."""
    def progress_callback(message):
        self.update_state(state='PROGRESS', meta={'status': message})
    try:
        # Fetch portfolio details from the database
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            cursor = conn.cursor()
            res = cursor.execute("SELECT stocks_json FROM custom_portfolios WHERE id = ?", (portfolio_id,)).fetchone()
            if not res:
                raise ValueError(f"Portfolio with ID {portfolio_id} not found.")
            holdings = json.loads(res[0])

        # We can also add the optimizer choice to custom portfolios in the future
        results = run_custom_portfolio_backtest(
            holdings=holdings, 
            start_date_str=start_date_str, 
            end_date_str=end_date_str, 
            risk_free_rate=risk_free_rate, 
            progress_callback=progress_callback,
            universe_name=universe_name # Pass it down
        )
        return results
    except Exception as e:
        traceback.print_exc()
        raise e

@shared_task(bind=True)
def run_live_analysis_task(self, universe_name, top_n, risk_free_rate, optimization_method, sector_constraints=None):
    """Celery task for performing the heavy live analysis and optimization."""
    try:
        # Load the model inside the task
        model_path = 'app/stock_selector_model.joblib' # Define path inside the task
        stock_model = joblib.load(model_path)
        
        self.update_state(state='PROGRESS', meta={'status': 'Ranking stocks with ML model...'})
        symbols_in_universe = data_fetcher.get_stock_universe(universe_name)
        top_stocks, stock_rankings_raw, feature_importances = ml_models.predict_top_stocks(stock_model, symbols_in_universe, top_n)
        
        if not top_stocks:
            # Use a dictionary to signal failure with a reason
            return {'error': f'ML model did not return any stock picks for the {universe_name} universe.'}

        self.update_state(state='PROGRESS', meta={'status': 'Optimizing portfolio...'})
        portfolio_data = ml_models.get_portfolio_data(top_stocks)
        
        if not portfolio_data:
            return {'error': 'Could not fetch portfolio data for the selected top stocks.'}

        if optimization_method == 'hrp':
            optimal_weights = ml_models.optimize_hrp_portfolio(portfolio_data, sector_constraints)        
        else:
            optimal_weights = ml_models.optimize_portfolio(portfolio_data, risk_free_rate)
            
        sector_exposure = ml_models.get_portfolio_sector_exposure(portfolio_data, optimal_weights)
        rationale = ml_models.generate_portfolio_rationale(optimal_weights, sector_exposure)
         # --- FIX #1: Create the simple, serializable stock-to-sector map ---
        # This replaces the need to return the non-serializable 'portfolio_data' object.
        stock_sector_map = {
            symbol: data['Sector'].iloc[0]
            for symbol, data in portfolio_data.items() if not data.empty and 'Sector' in data.columns
        }

        # --- FIX #2: Explicitly convert NumPy types in stock_rankings to native Python types ---
        # The scores from the ML model are numpy.float64, which can also cause JSON errors.
        stock_rankings_serializable = [(stock, float(score)) for stock, score in stock_rankings_raw]

        # --- FINAL STEP: Return a dictionary containing ONLY JSON-serializable data ---
        return {
            'top_stocks': top_stocks,
            'optimal_weights': optimal_weights,
            'sector_exposure': sector_exposure,
            'rationale': rationale,
            'stock_rankings': stock_rankings_serializable,  # Use the sanitized version
            'feature_importances': feature_importances,
            'stock_sector_map': stock_sector_map           # Use the new simple map
        }
    except Exception as e:
        traceback.print_exc()
        return {'error': f'An unexpected error occurred: {str(e)}'}