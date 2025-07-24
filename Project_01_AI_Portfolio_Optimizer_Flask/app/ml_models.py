import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pypfopt import HRPOpt
from .data_fetcher import get_historical_data
from datetime import date, timedelta
from .strategy import generate_all_features
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import plotly.figure_factory as ff
from scipy.cluster import hierarchy as sch
import json

def predict_top_stocks(model, symbols, top_n=10):
    if model is None:
        return [], {}, []

    end_date = date.today()
    start_date = end_date - timedelta(days=400)
    
    benchmark_df = get_historical_data('^NSEI', start_date, end_date)
    if benchmark_df.empty:
        print("--> ERROR: Could not fetch benchmark data for live analysis.")
        return [], {}, []

    predictions = {}
    feature_cols = [
        'MA_20', 'MA_50', 'ROC_20', 'Volatility_20D', 'RSI', 'Relative_Strength',
        'Momentum_3M', 'Momentum_6M', 'Momentum_12M', 'Sharpe_3M'
    ]

    for symbol in symbols:
        data = get_historical_data(symbol, start_date, end_date)
        if data.empty or len(data) < 252:
            continue

        all_features_df = generate_all_features(data, benchmark_df)
        
        if not all(col in all_features_df.columns for col in feature_cols):
            continue
            
        latest_features = all_features_df[feature_cols].dropna()

        if not latest_features.empty:
            prediction = model.predict(latest_features.tail(1))[0]
            predictions[symbol] = prediction
    
    if not predictions:
        return [], {}, []

    # Get the raw feature importances from the model
    raw_importances = sorted(zip(model.feature_name_, model.feature_importances_), key=lambda x: x[1], reverse=True)

    # --- THIS IS THE CRITICAL FIX ---
    # Convert the NumPy int32/int64 types to standard Python int types
    # so they can be serialized to JSON without errors.
    feature_importances = [(name, int(importance)) for name, importance in raw_importances]
    # --- END OF FIX ---

    sorted_stocks_with_scores = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    
    top_stock_names = [stock[0] for stock in sorted_stocks_with_scores[:top_n]]
    
    return top_stock_names, sorted_stocks_with_scores, feature_importances

def get_portfolio_data(symbols):
    end_date = date.today()
    start_date = end_date - timedelta(days=400)
    portfolio_data = {}
    for symbol in symbols:
        data = get_historical_data(symbol, start_date, end_date)
        if not data.empty:
            portfolio_data[symbol] = data
    return portfolio_data

def optimize_portfolio(portfolio_data, risk_free_rate):
    symbols = list(portfolio_data.keys())
    if len(symbols) < 2: return {symbols[0]: 1.0} if symbols else {}
    close_prices = {symbol: data['Close'] for symbol, data in portfolio_data.items()}
    portfolio_df = pd.DataFrame(close_prices).ffill().bfill()
    returns = portfolio_df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(symbols)
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        p_ret = np.sum(mean_returns * weights) * 252
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return -(p_ret - risk_free_rate) / p_std if p_std != 0 else -np.inf
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    max_weight = 0.10
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    result = minimize(neg_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    weights[weights < 0.001] = 0
    weights /= np.sum(weights)
    return dict(zip(symbols, np.round(weights, 4)))


def optimize_hrp_portfolio(portfolio_data, sector_constraints=None):
    symbols = list(portfolio_data.keys())
    if sector_constraints:
        print("--- Running HRP with Sector Constraints ---")
        return optimize_hrp_with_sector_constraints(portfolio_data, sector_constraints)
    
    # --- This is the original HRP logic for when no constraints are provided ---
    print("--- Running Standard HRP ---")
    symbols = list(portfolio_data.keys())
    if len(symbols) < 2: return {symbols[0]: 1.0} if symbols else {}
    close_prices = {symbol: data['Close'] for symbol, data in portfolio_data.items()}
    portfolio_df = pd.DataFrame(close_prices).ffill().bfill()
    returns = portfolio_df.pct_change().dropna()
    # Fallback to equal weight if there's not enough data for HRP
    if returns.empty or len(returns) < 2:
        print("Warning: Not enough return data for HRP. Falling back to equal weight.")
        return {symbol: 1.0 / len(symbols) for symbol in symbols}
    # 1. Get initial weights from the standard HRP algorithm
    hrp = HRPOpt(returns)
    hrp_weights = hrp.optimize()
    # 2. Enforce the 10% maximum weight constraint
    max_weight = 0.10
    weights = pd.Series(hrp_weights)
    # Find stocks that are overweight and calculate the total excess weight
    overweight_stocks = weights[weights > max_weight]
    excess_weight = (overweight_stocks - max_weight).sum()
    # Cap the overweight stocks at the maximum weight
    weights[overweight_stocks.index] = max_weight
    # 3. Redistribute the excess weight proportionally to the stocks under the cap
    underweight_stocks = weights[weights < max_weight]
    if not underweight_stocks.empty and excess_weight > 0:
        # The sum of underweight stocks is now the base for proportional distribution
        total_underweight_value = underweight_stocks.sum()
        weights[underweight_stocks.index] += excess_weight * (underweight_stocks / total_underweight_value)
    # 4. Final normalization to ensure the sum is exactly 1 due to potential floating point inaccuracies
    final_weights = weights / weights.sum()
    return {k: round(v, 4) for k, v in final_weights.items()}

def get_portfolio_sector_exposure(portfolio_data, weights):
    sector_exposure = {}
    for symbol, weight in weights.items():
        if symbol in portfolio_data and not portfolio_data[symbol].empty:
            sector = portfolio_data[symbol]['Sector'].iloc[0]
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
    return sector_exposure

def generate_portfolio_rationale(weights, sector_exposure):
    if not weights or not sector_exposure:
        return "<p class='text-danger'>Could not generate rationale due to insufficient data.</p>"
    top_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    top_sectors = sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True)[:2]
    holdings_str = ", ".join([f"<strong>{s} ({w*100:.1f}%)</strong>" for s, w in top_holdings if w > 0])
    sectors_str = ", ".join([f"<strong>{s} ({w*100:.1f}%)</strong>" for s, w in top_sectors if w > 0])
    rationale = f"""
    <h4>Portfolio Rationale:</h4>
    <p>This portfolio has been constructed by selecting stocks with the highest predicted forward returns from our ML model, and then optimizing their weights according to the chosen methodology.</p>
    <h5>Key Characteristics:</h5>
    <ul>
        <li><strong>Primary Holdings:</strong> The portfolio is led by {holdings_str}.</li>
        <li><strong>Sector Concentration:</strong> The allocation is primarily focused on the {sectors_str} sectors.</li>
    </ul>
    <p class="text-muted small"><em>Disclaimer: This is an AI-generated analysis for illustrative purposes. Not investment advice.</em></p>
    """
    return rationale

# === ADD THIS ENTIRE NEW FUNCTION ===
def optimize_hrp_with_sector_constraints(portfolio_data, sector_constraints, max_stock_weight=0.10):
    """
    Performs a two-stage optimization:
    1. Allocates weights to sectors based on user constraints and variance minimization.
    2. Runs HRP on stocks within each sector to determine final weights.
    """
    if not portfolio_data or len(portfolio_data) < 2:
        return {list(portfolio_data.keys())[0]: 1.0} if portfolio_data else {}

    # --- Pre-computation: Group stocks and calculate returns ---
    returns_df = pd.DataFrame({
        symbol: data['Close'].pct_change() for symbol, data in portfolio_data.items()
    }).dropna()

    sectors = {symbol: data['Sector'].iloc[0] for symbol, data in portfolio_data.items()}
    unique_sectors = sorted(list(set(sectors.values())))
    stocks_by_sector = {sec: [s for s, s_sec in sectors.items() if s_sec == sec] for sec in unique_sectors}

    # --- Stage 1: Sector-Level Optimization ---
    # Create a covariance matrix of the sectors themselves
    sector_returns = pd.DataFrame({
        sec: returns_df[stocks].mean(axis=1) for sec, stocks in stocks_by_sector.items()
    })
    sector_cov_matrix = sector_returns.cov()

    def objective(weights):
        # Objective is to minimize portfolio variance at the sector level
        return weights.T @ sector_cov_matrix.values @ weights

    # Initial guess: equal weight for each sector
    initial_weights = np.array([1/len(unique_sectors)] * len(unique_sectors))
    
    # Constraints:
    # 1. The sum of all sector weights must be 1.
    cons = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
    # 2. Bounds for each sector (min/max constraints from user)
    bounds = []
    for sector_name in unique_sectors:
        min_w = sector_constraints.get(sector_name, {}).get('min', 0) / 100
        max_w = sector_constraints.get(sector_name, {}).get('max', 100) / 100
        if min_w > max_w:
            print(f"Warning: Invalid constraint for {sector_name} (min {min_w*100}% > max {max_w*100}%). Ignoring max constraint.")
            max_w = 1.0 # Reset max to 100%
        bounds.append((min_w, max_w))

    # Run the sector-level optimizer
    sector_optim_result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)
    optimal_sector_weights = pd.Series(sector_optim_result.x, index=unique_sectors)

    # --- Stage 2: Intra-Sector HRP and Final Weight Calculation ---
    final_stock_weights = {}
    for sector, sector_weight in optimal_sector_weights.items():
        if sector_weight < 1e-5:  # Ignore sectors with negligible weight
            continue
        
        sector_stocks = stocks_by_sector[sector]
        sector_returns_df = returns_df[sector_stocks]
        
        if len(sector_stocks) > 1:
            hrp = HRPOpt(sector_returns_df)
            intra_sector_weights = hrp.optimize()
        else:
            # If only one stock in a sector, it gets the full sector weight
            intra_sector_weights = {sector_stocks[0]: 1.0}
            
        # Distribute the sector's total weight among its stocks
        for stock, hrp_weight in intra_sector_weights.items():
            final_stock_weights[stock] = sector_weight * hrp_weight

    # --- Final Step: Apply the 10% max individual stock weight constraint ---
    weights = pd.Series(final_stock_weights)
    overweight_stocks = weights[weights > max_stock_weight]
    if not overweight_stocks.empty:
        excess_weight = (overweight_stocks - max_stock_weight).sum()
        weights[overweight_stocks.index] = max_stock_weight
        underweight_stocks = weights[weights < max_stock_weight]
        if not underweight_stocks.empty and excess_weight > 0:
            weights[underweight_stocks.index] += excess_weight * (underweight_stocks / underweight_stocks.sum())

    # Normalize to ensure sum is exactly 1 and return
    final_weights = weights / weights.sum()
    return {k: round(v, 4) for k, v in final_weights.items()}
def analyze_portfolio_clusters(portfolio_data):
    """
    Performs hierarchical clustering on a portfolio, generates a dendrogram,
    identifies both main and tight clusters, and gathers the latest features for each stock.
    """
    if not portfolio_data or len(portfolio_data) < 2:
        return {"error": "Not enough data to perform cluster analysis."}

    # --- 1. Robust Data Cleaning ---
    returns_df = pd.DataFrame({
        symbol: data['Close'].pct_change() for symbol, data in portfolio_data.items()
    })
    returns_df.fillna(0, inplace=True)

    # Remove stocks with near-zero variance to prevent clustering issues
    variances = returns_df.var()
    valid_stocks = variances[variances > 1e-10].index.tolist()
    
    if len(valid_stocks) < len(returns_df.columns):
        dropped_stocks = set(returns_df.columns) - set(valid_stocks)
        print(f"Warning: Dropping stocks with zero variance for clustering: {list(dropped_stocks)}")
    
    if len(valid_stocks) < 2:
        return {"error": "Not enough stocks with valid variance for clustering."}
    
    clean_returns_df = returns_df[valid_stocks]

    # --- 2. Perform Clustering ---
    corr_matrix = clean_returns_df.corr()
    labels = corr_matrix.index.tolist()  # Consistent list of stock symbols
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    dist_matrix.fillna(0, inplace=True)
    
    # Convert to condensed distance matrix and clean non-finite values
    condensed_dist_matrix = squareform(dist_matrix, checks=False)
    condensed_dist_matrix = np.nan_to_num(condensed_dist_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Perform hierarchical clustering with single linkage
    Z = linkage(condensed_dist_matrix, 'single')

    # --- 3. Generate Dendrogram and Map Labels ---
    # Create dendrogram without labels to avoid dimension mismatch
    fig = ff.create_dendrogram(Z, orientation='left')
    
    # Get the correct order of leaves from SciPy's dendrogram
    dendro_data = sch.dendrogram(Z, no_plot=True)
    leaf_indices = dendro_data['leaves']
    ordered_labels = [labels[i] for i in leaf_indices]
    
    # Update y-axis with ordered stock symbols
    fig.update_layout(yaxis=dict(tickmode='array', tickvals=fig.layout.yaxis.tickvals, ticktext=ordered_labels))
    
    # Customize layout for better visualization
    fig.update_layout(
        height=max(600, len(labels) * 25),  # Adjust height based on number of stocks
        margin=dict(l=150, r=20, t=40, b=20),
        title_text='Hierarchical Stock Clusters (based on Correlation)'
    )
    dendrogram_json = json.loads(fig.to_json())

    # --- 4. Identify Both Main and Tight Clusters ---
    # Tight clusters (for redundancy analysis, high correlation)
    tight_cluster_indices = fcluster(Z, t=0.3, criterion='distance')
    tight_clusters_map = {}
    for i, stock_name in enumerate(labels):
        cluster_id = tight_cluster_indices[i]
        if cluster_id not in tight_clusters_map:
            tight_clusters_map[cluster_id] = []
        tight_clusters_map[cluster_id].append(stock_name)
    tight_clusters = {f"Redundant Group {cid}": stocks for cid, stocks in tight_clusters_map.items() if len(stocks) > 1}

    # Main clusters (high-level structural groups, up to 4 clusters)
    num_main_clusters = min(len(labels) - 1, 4)  # Aim for 4 clusters, limited by N-1
    main_cluster_indices = fcluster(Z, t=num_main_clusters, criterion='maxclust')
    main_clusters_map = {}
    for i, stock_name in enumerate(labels):
        cluster_id = main_cluster_indices[i]
        if cluster_id not in main_clusters_map:
            main_clusters_map[cluster_id] = []
        main_clusters_map[cluster_id].append(stock_name)
    main_clusters = {f"Main Cluster {cid}": stocks for cid, stocks in main_clusters_map.items()}

    # --- 5. Gather Latest Features for AI Analysis ---
    feature_data = {}
    end_date = date.today()
    start_date = end_date - timedelta(days=400)
    benchmark_df = get_historical_data('^NSEI', start_date, end_date)
    feature_cols_for_ai = ['ROC_20', 'Volatility_20D', 'Sharpe_3M', 'RSI']

    for symbol in labels:
        data = portfolio_data.get(symbol)
        if data is None or data.empty or benchmark_df.empty:
            continue
        
        all_features_df = generate_all_features(data, benchmark_df)
        latest_features = all_features_df[feature_cols_for_ai].dropna().tail(1)
        
        if not latest_features.empty:
            feature_data[symbol] = latest_features.to_dict('records')[0]

    return {
        "dendrogram_json": dendrogram_json,
        "tight_clusters": tight_clusters,
        "main_clusters": main_clusters,  # Include main clusters in the response
        "feature_data": feature_data
    }