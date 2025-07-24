import pandas as pd
import numpy as np
import quantstats as qs
from tqdm import tqdm
from datetime import date
from dateutil.relativedelta import relativedelta
import json
import lightgbm as lgb

from . import factor_analysis
from .data_fetcher import get_stock_universe, get_historical_data
from .ml_models import optimize_portfolio, optimize_hrp_portfolio, get_portfolio_sector_exposure
from .strategy import generate_all_features
from .reporting import generate_gemini_report

def to_json_safe(obj):
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, pd.Index): return obj.tolist()
    if pd.isna(obj): return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def generate_report_payload(portfolio_returns, benchmark_returns, holdings_df, master_raw_data, rebalance_logs, risk_free_rate):
    if holdings_df.sum().sum() < 1e-9:
        print("--- [Reporting] No trades were made. Generating a zero-performance report. ---")
        start_date = holdings_df.index.min()
        end_date = holdings_df.index.max()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        strategy_equity = pd.Series(1.0, index=date_range)
        benchmark_equity = (1 + benchmark_returns.fillna(0)).cumprod()

        kpis = {
            "CAGR﹪": 0.0, "Sharpe": 0.0, "Max Drawdown": 0.0,
            "Sortino": 0.0, "Beta": 0.0, "Calmar": 0.0,
            "Daily VaR": 0.0, "Daily CVaR": 0.0,
            "Error": "Strategy did not execute any trades. This may be due to the regime filter always being active or the model never producing positive predictions."
        }
        
        return {
            "kpis": kpis,
            "factor_exposure": {"error": "Factor analysis not run because no trades were made."},
            "charts": {
                "equity": { "data": [
                    {'x': strategy_equity.index.strftime('%Y-%m-%d').tolist(), 'y': strategy_equity.values.tolist(), 'mode': 'lines', 'name': 'Strategy (No Trades)'},
                    {'x': benchmark_equity.index.strftime('%Y-%m-%d').tolist(), 'y': benchmark_equity.values.tolist(), 'mode': 'lines', 'name': 'Benchmark (NIFTY 50)'}
                ], "layout": {'title': 'Strategy vs. Benchmark Performance'} },
                "drawdown": {"data": [], "layout": {'title': 'Strategy Drawdowns (No Trades)'}},
                "historical_weights": {"data": [], "layout": {'title': 'Historical Stock Weights (No Trades)'}},
                "historical_sectors": {"data": [], "layout": {'title': 'Historical Sector Exposure (No Trades)'}}
            },
            "tables": {"monthly_returns": "{}", "yearly_returns": "{}"},
            "logs": rebalance_logs,
            "ai_report": "AI analysis skipped: The strategy did not make any trades."
        }

    portfolio_returns.fillna(0, inplace=True)
    benchmark_returns.fillna(0, inplace=True)
    combined = pd.merge(portfolio_returns.rename('Strategy'), benchmark_returns.rename('Benchmark'), left_index=True, right_index=True, how='inner')
    
    portfolio_returns_clean = combined['Strategy']
    benchmark_returns_clean = combined['Benchmark']
    
    kpis_df = qs.reports.metrics(portfolio_returns_clean, benchmark=benchmark_returns_clean, rf=risk_free_rate, display=False)
    strategy_kpis = kpis_df.loc[:, 'Strategy']
    benchmark_kpis = kpis_df.loc[:, 'Benchmark'] if 'Benchmark' in kpis_df.columns else pd.Series(dtype='float64')


    
    factor_exposure_results = factor_analysis.analyze_factor_exposure(portfolio_returns_clean)
    rolling_factor_results_json = factor_analysis.analyze_rolling_factor_exposure(portfolio_returns_clean)
    drawdown_series = qs.stats.to_drawdown_series(portfolio_returns_clean)
    monthly_returns_df = qs.stats.monthly_returns(portfolio_returns_clean, compounded=True)
    yearly_returns_df = portfolio_returns_clean.resample('YE').apply(lambda x: (1 + x).prod() - 1).to_frame(name='Strategy')
    yearly_returns_df.index = yearly_returns_df.index.year

    strategy_equity = (1 + portfolio_returns_clean).cumprod()
    benchmark_equity = (1 + benchmark_returns_clean).cumprod()
    
    sector_exposure_over_time = {}
    for a_date, weights in holdings_df.iterrows():
        portfolio_data = { s: master_raw_data[s] for s in weights.index if s in master_raw_data and weights.get(s, 0) > 0 }
        sector_exposure_over_time[a_date] = get_portfolio_sector_exposure(portfolio_data, weights)
    sector_exposure_df = pd.DataFrame.from_dict(sector_exposure_over_time, orient='index').fillna(0)
    
    stock_traces = [{'x': holdings_df.index.strftime('%Y-%m-%d').tolist(), 'y': (holdings_df[stock] * 100).tolist(), 'name': stock, 'type': 'bar'} for stock in holdings_df.columns if holdings_df[stock].sum() > 0]
    stock_layout = {'title': 'Historical Stock Weights (%)', 'barmode': 'stack', 'yaxis': {'ticksuffix': '%'}, 'legend': {'traceorder': 'reversed'}}
    
    sector_traces = [{'x': sector_exposure_df.index.strftime('%Y-%m-%d').tolist(), 'y': (sector_exposure_df[sector] * 100).tolist(), 'name': sector, 'type': 'bar'} for sector in sector_exposure_df.columns if sector_exposure_df[sector].sum() > 0]
    sector_layout = {'title': 'Historical Sector Exposure (%)', 'barmode': 'stack', 'yaxis': {'ticksuffix': '%'}, 'legend': {'traceorder': 'reversed'}}

    ai_report = generate_gemini_report(strategy_kpis.to_dict(), {}, yearly_returns_df['Strategy'].to_dict(), rebalance_logs)
    
    results_payload = {
        "kpis": strategy_kpis.to_dict(),
        "benchmark_kpis": benchmark_kpis.to_dict(),
        "factor_exposure": factor_exposure_results,
        "charts": {
            "equity": { "data": [{'x': strategy_equity.index.strftime('%Y-%m-%d').tolist(), 'y': strategy_equity.values.tolist(), 'mode': 'lines', 'name': 'Strategy', 'line': {'color': '#0d6efd', 'width': 2}}, {'x': benchmark_equity.index.strftime('%Y-%m-%d').tolist(), 'y': benchmark_equity.values.tolist(), 'mode': 'lines', 'name': 'Benchmark (NIFTY 50)', 'line': {'color': '#6c757d', 'dash': 'dot', 'width': 1.5}}], "layout": {'title': 'Strategy vs. Benchmark Performance', 'yaxis': {'title': 'Cumulative Growth', 'type': 'log'}, 'legend': {'x': 0.01, 'y': 0.99}, 'margin': {'t': 40, 'b': 40, 'l': 60, 'r': 20}} },
            "drawdown": { "data": [{'x': drawdown_series.index.strftime('%Y-%m-%d').tolist(), 'y': (drawdown_series.values * 100).tolist(), 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy', 'name': 'Drawdown', 'line': {'color': '#dc3545'}}], "layout": {'title': 'Strategy Drawdowns', 'yaxis': {'title': 'Drawdown (%)'}, 'margin': {'t': 40, 'b': 40, 'l': 60, 'r': 20}} },
            "historical_weights": {"data": stock_traces, "layout": stock_layout},
            "historical_sectors": {"data": sector_traces, "layout": sector_layout},
            "rolling_factor_betas": rolling_factor_results_json
        },
        "tables": { "monthly_returns": monthly_returns_df.to_json(orient='split'), "yearly_returns": yearly_returns_df.to_json(orient='split') },
        "logs": rebalance_logs,
        "ai_report": ai_report,   
    }
    return json.loads(json.dumps(results_payload, default=to_json_safe))

def calculate_performance(holdings_df, master_raw_data, start_date_str, end_date_str, risk_free_rate, rebalance_logs):
    log_progress = lambda message: print(message)
    log_progress("--- [Reporting] Starting performance calculation...")

    clean_date_index = pd.date_range(start=start_date_str, end=end_date_str, freq='B')
    
    benchmark_data = get_historical_data('^NSEI', start_date_str, end_date_str)
    benchmark_returns = benchmark_data['Close'].pct_change(fill_method=None)
    benchmark_returns.name = 'Benchmark'
    
    if holdings_df.sum().sum() < 1e-9:
        return generate_report_payload(pd.Series(), benchmark_returns, holdings_df, master_raw_data, rebalance_logs, risk_free_rate)

    valid_cols = [col for col in holdings_df.columns if col in master_raw_data]
    price_df = pd.DataFrame({
        symbol: master_raw_data[symbol]['Close'] for symbol in valid_cols
    }).reindex(clean_date_index, method='ffill')
    
    price_df.sort_index(inplace=True)
    returns_df = price_df.pct_change(fill_method=None)
    
    holdings_df.sort_index(inplace=True)
    aligned_holdings = holdings_df.reindex(returns_df.index, method='ffill').fillna(0)

    TRANSACTION_COST_BPS = 15
    turnover = (aligned_holdings.shift(1).fillna(0) - aligned_holdings).abs().sum(axis=1) / 2
    transaction_costs = turnover * (TRANSACTION_COST_BPS / 10000)
    portfolio_returns = (aligned_holdings * returns_df[aligned_holdings.columns]).sum(axis=1) - transaction_costs
    portfolio_returns.name = 'Strategy'
    
    return generate_report_payload(portfolio_returns, benchmark_returns, holdings_df, master_raw_data, rebalance_logs, risk_free_rate)

# In app/backtesting.py

# Replace the entire run_backtest function with this new version.
def run_backtest(start_date_str, end_date_str, universe_name, top_n, risk_free_rate, rebalance_freq='BMS', progress_callback=None, optimization_method='hrp', sector_constraints=None):
    def log_progress(message):
        if progress_callback: progress_callback(message)

    log_progress("--- [Backtest Engine] Initializing ML Walk-Forward Backtest ---")
    
    all_symbols = get_stock_universe(universe_name)
    earliest_date = pd.to_datetime(start_date_str) - relativedelta(years=5) # 5 years for features + training buffer

    log_progress("Loading all historical data for the universe...")
    master_raw_data = {
        symbol: get_historical_data(symbol, earliest_date, end_date_str)
        for symbol in tqdm(all_symbols, desc="Loading Stock Data")
    }
    master_raw_data = {k: v for k, v in master_raw_data.items() if not v.empty and len(v) > 252}
    
    benchmark_master_df = get_historical_data('^NSEI', earliest_date, end_date_str)
    if benchmark_master_df.empty:
        raise ValueError("Could not load master benchmark data. Backtest cannot proceed.")

    # === PERFORMANCE OPTIMIZATION: PRE-COMPUTE ALL FEATURES ONCE ===
    log_progress("Pre-computing all features for all stocks...")
    master_features = {}
    feature_cols = ['MA_20', 'MA_50', 'ROC_20', 'Volatility_20D', 'RSI', 'Relative_Strength', 'Momentum_3M', 'Momentum_6M', 'Momentum_12M', 'Sharpe_3M']
    
    for symbol, raw_data in tqdm(master_raw_data.items(), desc="Generating Features"):
        features_df = generate_all_features(raw_data, benchmark_master_df)
        master_features[symbol] = features_df.dropna(subset=feature_cols)

    log_progress("--- Starting Walk-Forward Simulation... ---")
    rebalance_dates = pd.date_range(start=start_date_str, end=end_date_str, freq=rebalance_freq)
    all_holdings = {}
    rebalance_logs = []
    model = None
    last_train_date = pd.Timestamp.min

    for rebalance_date in tqdm(rebalance_dates, desc="Backtesting Progress"):
        # --- MODEL RETRAINING LOGIC (NOW MUCH FASTER) ---
        if model is None or (rebalance_date - last_train_date).days > 365:
            log_progress(f"--- Retraining model for date: {rebalance_date.date()} ---")
            train_start = rebalance_date - relativedelta(years=3)
            train_end = rebalance_date
            
            # SLICE the pre-computed features instead of recalculating
            training_data_slices = [
                df.loc[train_start:train_end] 
                for df in master_features.values() 
                if not df.loc[train_start:train_end].empty
            ]

            if training_data_slices:
                full_dataset = pd.concat(training_data_slices).dropna(subset=['Target'])
                if not full_dataset.empty:
                    X_train, y_train = full_dataset[feature_cols], full_dataset['Target']
                    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, n_jobs=-1, random_state=42)
                    model.fit(X_train, y_train)
                    last_train_date = rebalance_date
                    log_progress("--- Model retraining complete. ---")
                else:
                    log_progress("--- Not enough target data in slice for retraining, using previous model. ---")
            else:
                log_progress("--- Not enough data in slice for retraining, using previous model. ---")

        # --- The rest of the backtesting loop remains the same ---
        current_log = {'Date': rebalance_date.strftime('%Y-%m-%d'), 'Action': 'Hold Cash', 'Details': {}}
        
        # Regime Filter...
        try:
            nifty_past_data = benchmark_master_df.loc[benchmark_master_df.index < rebalance_date]
            if len(nifty_past_data) < 200:
                current_log['Details'] = "Not enough market history for 200-day MA."; all_holdings[rebalance_date] = {}; rebalance_logs.append(current_log); continue
            last_price = nifty_past_data['Close'].iloc[-1]
            nifty_ma_200 = nifty_past_data['Close'].rolling(window=200).mean().iloc[-1]
            if last_price < nifty_ma_200:
                current_log['Details'] = f"Regime filter: NIFTY Close ({last_price:.2f}) < 200-MA ({nifty_ma_200:.2f})"; all_holdings[rebalance_date] = {}; rebalance_logs.append(current_log); continue
        except (IndexError, ValueError):
            current_log['Details'] = "Error in regime filter."; all_holdings[rebalance_date] = {}; rebalance_logs.append(current_log); continue

        if model is None:
            current_log['Details'] = "ML model not trained yet."; all_holdings[rebalance_date] = {}; rebalance_logs.append(current_log); continue

        # Predictions...
        predictions = {}
        for symbol, features_df in master_features.items():
            features_at_date = features_df.loc[features_df.index < rebalance_date]
            if not features_at_date.empty:
                predictions[symbol] = model.predict(features_at_date.tail(1)[feature_cols])[0]
        
        top_stocks = [s for s, p in sorted(predictions.items(), key=lambda item: item[1], reverse=True) if p > 0][:top_n]
        if not top_stocks:
            current_log['Details'] = "No stocks had positive predictions."; all_holdings[rebalance_date] = {}; rebalance_logs.append(current_log); continue

        # Optimization...
        portfolio_data = {s: master_raw_data[s].loc[master_raw_data[s].index < rebalance_date] for s in top_stocks}
        if len(portfolio_data) >= 2:
            log_progress(f"Optimizing portfolio for {rebalance_date.date()}...")
            if optimization_method == 'hrp':
                weights = optimize_hrp_portfolio(portfolio_data, sector_constraints)
            else:
                weights = optimize_portfolio(portfolio_data, risk_free_rate)
            all_holdings[rebalance_date] = weights
            current_log['Action'] = 'Rebalanced Portfolio'
            current_log['Details'] = weights
        else:
            current_log['Details'] = "Not enough valid stocks to form portfolio."
            all_holdings[rebalance_date] = {}
        rebalance_logs.append(current_log)

    holdings_df = pd.DataFrame.from_dict(all_holdings, orient='index').fillna(0)
    return calculate_performance(holdings_df, master_raw_data, start_date_str, end_date_str, risk_free_rate, rebalance_logs)
def run_custom_portfolio_backtest(holdings, start_date_str, end_date_str, risk_free_rate, universe_name, rebalance_freq='BMS', progress_callback=None):
    """
    Backtests a custom portfolio, filtering its stocks against a selected universe,
    and then dynamically rebalancing using HRP.
    """
    def log_progress(message):
        if progress_callback: progress_callback(message)

    log_progress("--- [Custom Backtest] Initializing Dynamic HRP Backtest ---")
    
    # --- NEW LOGIC: Filter portfolio stocks against the selected universe ---
    
    # 1. Get the list of stocks from the user's saved custom portfolio
    portfolio_symbols = list(holdings.keys())
    
    # 2. Get the list of stocks from the selected backtest universe
    log_progress(f"Fetching symbols for universe: '{universe_name}'...")
    universe_symbols = get_stock_universe(universe_name)
    
    # 3. Find the intersection: these are the stocks we will actually backtest.
    #    This ensures we only consider stocks present in BOTH the portfolio AND the universe.
    final_symbols_to_backtest = list(set(portfolio_symbols) & set(universe_symbols))
    
    # 4. Handle the case where there is no overlap
    if not final_symbols_to_backtest:
        log_progress(f"CRITICAL: No stocks from the portfolio exist in the selected universe '{universe_name}'. The backtest will show zero trades.")
        # Create an empty holdings dataframe to signal that no trades were made.
        # This will correctly generate a flat performance report.
        empty_holdings_df = pd.DataFrame(columns=portfolio_symbols, index=pd.to_datetime([]))
        # The rebalance logs will be empty in this case.
        rebalance_logs = [
            {'Date': start_date_str, 'Action': 'Hold Cash', 'Details': 'No overlapping stocks between portfolio and selected universe.'}
        ]
        return calculate_performance(empty_holdings_df, {}, start_date_str, end_date_str, risk_free_rate, rebalance_logs)

    log_progress(f"Found {len(final_symbols_to_backtest)} overlapping stocks to backtest.")
    # --- END OF NEW LOGIC ---

    # The rest of the function now operates only on the `final_symbols_to_backtest`
    earliest_date = pd.to_datetime(start_date_str) - relativedelta(years=3) # Need 3 years history for HRP

    log_progress(f"Loading data for {len(final_symbols_to_backtest)} filtered stocks...")
    master_raw_data = {
        symbol: get_historical_data(symbol, earliest_date, end_date_str)
        for symbol in tqdm(final_symbols_to_backtest, desc="Loading Filtered Portfolio Data")
    }
    master_raw_data = {k: v for k, v in master_raw_data.items() if not v.empty}
    
    log_progress("--- Simulating dynamic HRP rebalancing... ---")
    rebalance_dates = pd.date_range(start=start_date_str, end=end_date_str, freq=rebalance_freq)
    
    all_holdings = {}
    rebalance_logs = []
    
    for rebalance_date in tqdm(rebalance_dates, desc="Rebalancing Custom Portfolio"):
        # Get historical data for all stocks in the universe up to the rebalance date
        portfolio_data_slice = {
            s: data.loc[data.index < rebalance_date] 
            for s, data in master_raw_data.items()
        }
        
        # Filter out any stocks that don't have enough valid return data for this period
        valid_portfolio_data = {
            s: d for s, d in portfolio_data_slice.items() 
            if not d.empty and len(d['Close'].pct_change().dropna()) > 1
        }
        
        current_log = {'Date': rebalance_date.strftime('%Y-%m-%d')}
        
        if len(valid_portfolio_data) >= 2:
            # Perform HRP optimization on the available stocks for this period
            weights = optimize_hrp_portfolio(valid_portfolio_data)
            all_holdings[rebalance_date] = weights
            current_log['Action'] = 'Rebalanced to HRP Weights'
            current_log['Details'] = weights
        else:
            # If not enough stocks have data, hold cash for this period
            all_holdings[rebalance_date] = {}
            current_log['Action'] = 'Hold Cash'
            current_log['Details'] = 'Not enough assets with sufficient historical data to perform HRP optimization for this period.'

        rebalance_logs.append(current_log)
        
    holdings_df = pd.DataFrame.from_dict(all_holdings, orient='index').fillna(0)
    
    # The final step is to calculate and return the full performance report
    return calculate_performance(holdings_df, master_raw_data, start_date_str, end_date_str, risk_free_rate, rebalance_logs)
 