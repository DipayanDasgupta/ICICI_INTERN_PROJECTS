import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
from datetime import datetime, timedelta
import logging # Added for logging best practices

# It's good practice to get a logger instance
logger = logging.getLogger(__name__)

class RiskCalculator:
    # FIX: The __init__ method is updated to accept the 'symbols' and 'market_index' arguments.
    def __init__(self, symbols, market_index='^NSEI'):
        """
        Initializes the RiskCalculator.

        Args:
            symbols (list): A list of stock tickers to analyze.
            market_index (str): The ticker for the market index (default: Nifty 50).
        """
        self.symbols = symbols
        self.market_index = market_index
        self.confidence_levels = [0.95, 0.99]  # 95% and 99% confidence levels
        logger.info(f"RiskCalculator initialized for {len(symbols)} symbols against market index {market_index}")


    def calculate_var(self, returns, confidence_level=0.95, time_horizon=1):
        """Calculate Value at Risk (VaR)"""
        try:
            if returns.empty:
                return None
            
            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Handle case where returns has fewer elements than needed for index
            if len(sorted_returns) <= 1:
                return None
            
            # Calculate percentile
            index = int((1 - confidence_level) * len(sorted_returns))
            var = sorted_returns[index]
            
            # Scale for time horizon
            var_scaled = var * np.sqrt(time_horizon)
            
            return {
                'var': var_scaled,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
        
        except Exception as e:
            st.error(f"Error calculating VaR: {str(e)}")
            return None
    
    def calculate_cvar(self, returns, confidence_level=0.95, time_horizon=1):
        """Calculate Conditional Value at Risk (CVaR) - Expected Shortfall"""
        try:
            if returns.empty:
                return None

            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Calculate cutoff index
            index = int((1 - confidence_level) * len(sorted_returns))
            if index == 0: # Not enough data for CVaR at this confidence level
                return None

            # CVaR is the mean of returns below VaR
            cvar = np.mean(sorted_returns[:index])
            
            # Scale for time horizon
            cvar_scaled = cvar * np.sqrt(time_horizon)
            
            return {
                'cvar': cvar_scaled,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
        
        except Exception as e:
            st.error(f"Error calculating CVaR: {str(e)}")
            return None
    
    def calculate_portfolio_risk_metrics(self, portfolio_data, returns_data):
        """Calculate comprehensive risk metrics for the portfolio"""
        try:
            if returns_data.empty or not portfolio_data:
                return None
            
            # Calculate portfolio returns
            weights = []
            symbols = []
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            
            if total_value == 0:
                return None

            for symbol, data in portfolio_data.items():
                market_value = data.get('market_value', 0)
                weight = market_value / total_value
                weights.append(weight)
                symbols.append(symbol)
            
            weights = np.array(weights)
            
            # Calculate returns for symbols in portfolio
            portfolio_returns = []
            valid_symbols_in_returns = [s for s in symbols if s in returns_data.columns]

            for i, symbol in enumerate(valid_symbols_in_returns):
                symbol_returns = returns_data[symbol].pct_change().dropna()
                # Find the weight corresponding to the valid symbol
                weight_index = symbols.index(symbol)
                portfolio_returns.append(symbol_returns * weights[weight_index])
            
            if portfolio_returns:
                portfolio_returns_series = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            else:
                return None
            
            # Calculate risk metrics
            risk_metrics = {}
            
            # Basic statistics
            risk_metrics['volatility'] = portfolio_returns_series.std() * np.sqrt(252)  # Annualized
            risk_metrics['skewness'] = stats.skew(portfolio_returns_series.dropna())
            risk_metrics['kurtosis'] = stats.kurtosis(portfolio_returns_series.dropna())
            
            # VaR and CVaR
            for confidence_level in self.confidence_levels:
                var_result = self.calculate_var(portfolio_returns_series, confidence_level)
                cvar_result = self.calculate_cvar(portfolio_returns_series, confidence_level)
                
                if var_result and cvar_result:
                    risk_metrics[f'var_{int(confidence_level*100)}'] = var_result['var']
                    risk_metrics[f'cvar_{int(confidence_level*100)}'] = cvar_result['cvar']
            
            # Maximum Drawdown
            drawdown_result = self.calculate_max_drawdown(portfolio_returns_series)
            if drawdown_result:
                risk_metrics['max_drawdown'] = drawdown_result['max_drawdown']
            
            # Beta relative to market
            if self.market_index in returns_data.columns:
                nifty_returns = returns_data[self.market_index].pct_change().dropna()
                aligned_returns = pd.concat([portfolio_returns_series, nifty_returns], axis=1).dropna()
                if len(aligned_returns) > 30:  # Need sufficient data
                    covariance = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])[0, 1]
                    market_variance = np.var(aligned_returns.iloc[:, 1])
                    risk_metrics['beta'] = covariance / market_variance if market_variance != 0 else 0
                else:
                    risk_metrics['beta'] = None
            else:
                risk_metrics['beta'] = None
            
            # Tracking Error (relative to Nifty 50)
            if self.market_index in returns_data.columns:
                nifty_returns = returns_data[self.market_index].pct_change().dropna()
                aligned_returns = pd.concat([portfolio_returns_series, nifty_returns], axis=1).dropna()
                if len(aligned_returns) > 30:
                    tracking_error = (aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]).std() * np.sqrt(252)
                    risk_metrics['tracking_error'] = tracking_error
                else:
                    risk_metrics['tracking_error'] = None
            else:
                risk_metrics['tracking_error'] = None
            
            return risk_metrics
        
        except Exception as e:
            st.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return None
    
    def calculate_max_drawdown(self, returns_series):
        """Calculate maximum drawdown"""
        try:
            if returns_series.empty:
                return None
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_series).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = drawdown.min()
            
            # Find drawdown periods
            drawdown_start = None
            if not drawdown.empty:
                max_dd_end = drawdown.idxmin()
                
                # Find the start of the max drawdown period
                for i in range(len(cumulative_returns[:max_dd_end])):
                    if cumulative_returns.iloc[i] == running_max.loc[max_dd_end]:
                        drawdown_start = cumulative_returns.index[i]
            else:
                max_dd_end = None

            return {
                'max_drawdown': max_drawdown,
                'drawdown_series': drawdown,
                'drawdown_start': drawdown_start,
                'drawdown_end': max_dd_end,
                'recovery_date': None  # Would need to calculate when drawdown recovered
            }
        
        except Exception as e:
            st.error(f"Error calculating max drawdown: {str(e)}")
            return None
    
    def calculate_correlation_risk(self, returns_data, portfolio_data):
        """Calculate correlation-based risk metrics"""
        try:
            if returns_data.empty or not portfolio_data:
                return None
            
            # Get symbols in portfolio that are also in returns_data
            portfolio_symbols = [s for s in portfolio_data.keys() if s in returns_data.columns]
            
            if len(portfolio_symbols) < 2:
                return None # Not enough data for correlation

            # Filter returns data for portfolio symbols
            portfolio_returns = returns_data[portfolio_symbols].pct_change().dropna()
            
            if portfolio_returns.empty:
                return None
            
            # Calculate correlation matrix
            correlation_matrix = portfolio_returns.corr()
            
            # Calculate average correlation
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    correlations.append(correlation_matrix.iloc[i, j])
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Calculate concentration risk (Herfindahl-Hirschman Index)
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            weights = []
            if total_value > 0:
                for symbol in portfolio_symbols:
                    weight = portfolio_data[symbol].get('market_value', 0) / total_value
                    weights.append(weight)
            
            hhi = sum([w**2 for w in weights])
            
            # Risk assessment
            correlation_risk = "High" if avg_correlation > 0.7 else "Medium" if avg_correlation > 0.4 else "Low"
            concentration_risk = "High" if hhi > 0.25 else "Medium" if hhi > 0.15 else "Low"
            
            return {
                'correlation_matrix': correlation_matrix,
                'average_correlation': avg_correlation,
                'correlation_risk_level': correlation_risk,
                'concentration_index': hhi,
                'concentration_risk_level': concentration_risk,
                'diversification_ratio': 1 / hhi if hhi > 0 else 0
            }
        
        except Exception as e:
            st.error(f"Error calculating correlation risk: {str(e)}")
            return None
    
    def stress_test_portfolio(self, portfolio_data, returns_data, stress_scenarios=None):
        """Perform stress testing on the portfolio"""
        try:
            if not portfolio_data:
                return None
            
            # Calculate portfolio weights
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            if total_value == 0:
                return None

            weights = {}
            for symbol, data in portfolio_data.items():
                weight = data.get('market_value', 0) / total_value
                weights[symbol] = weight
            
            stress_results = {}
            
            # Default stress scenarios tailored for Indian market
            if not stress_scenarios:
                stress_scenarios = {
                    'Market Crash (-25%)': {'all_stocks': -0.25},
                    'IT Sector Selloff (-30%)': {'NIFTYIT.NS': -0.30},
                    'Banking Crisis (-20%)': {'BANKNIFTY.NS': -0.20},
                    'Rupee Depreciation (-10%)': {'all_stocks': -0.10}, # Simplified impact
                    'Monsoon Failure': {'NIFTYFMCG.NS': -0.20, 'NIFTYAUTO.NS':-0.15}
                }
            
            # A simple mapping of stocks to sectors for stress testing
            # In a real app, this would come from a more robust source
            stock_sector_map = {
                'RELIANCE.NS': 'NIFTY50', 'TCS.NS': 'NIFTYIT.NS', 'INFY.NS': 'NIFTYIT.NS',
                'HDFCBANK.NS': 'BANKNIFTY.NS', 'ICICIBANK.NS': 'BANKNIFTY.NS', 'SBIN.NS': 'BANKNIFTY.NS',
                'HINDUNILVR.NS': 'NIFTYFMCG.NS', 'DABUR.NS': 'NIFTYFMCG.NS', 'UPL.NS': 'NIFTY50'
            }

            for scenario_name, scenario_shocks in stress_scenarios.items():
                portfolio_shock = 0
                
                for symbol, weight in weights.items():
                    # Determine shock for the symbol
                    shock = 0
                    stock_sector = stock_sector_map.get(symbol)
                    
                    if symbol in scenario_shocks:
                        shock = scenario_shocks[symbol]
                    elif stock_sector in scenario_shocks:
                        shock = scenario_shocks[stock_sector]
                    elif 'all_stocks' in scenario_shocks:
                        shock = scenario_shocks['all_stocks']
                    
                    portfolio_shock += weight * shock
                
                stress_results[scenario_name] = {
                    'portfolio_impact': portfolio_shock,
                    'inr_impact': portfolio_shock * total_value,
                    'new_portfolio_value_inr': total_value * (1 + portfolio_shock)
                }
            
            return stress_results
        
        except Exception as e:
            st.error(f"Error performing stress test: {str(e)}")
            return None
    
    def calculate_risk_score(self, risk_metrics):
        """Calculate overall risk score (0-100, higher = riskier)"""
        try:
            if not risk_metrics:
                return None
            
            score = 0
            
            # Volatility component (0-30 points)
            vol = risk_metrics.get('volatility', 0.20) # Default to 20% if not available
            vol_score = min(30, (vol / 0.40) * 30)  # Normalize against a high 40% volatility
            score += vol_score
            
            # VaR component (0-25 points)
            var = abs(risk_metrics.get('var_95', 0.025)) # Default to 2.5% daily loss
            var_score = min(25, (var / 0.05) * 25) # Normalize against a high 5% VaR
            score += var_score
            
            # Max Drawdown component (0-25 points)
            dd = abs(risk_metrics.get('max_drawdown', 0.20)) # Default to 20% drawdown
            dd_score = min(25, (dd / 0.50) * 25) # Normalize against a high 50% drawdown
            score += dd_score
            
            # Beta component (0-20 points)
            beta = abs(risk_metrics.get('beta', 1.0)) # Default to market beta
            beta_score = min(20, ((beta - 0.5) / 1.5) * 20) # Normalize for beta between 0.5 and 2.0
            score += beta_score
            
            return min(100, max(0, score))
        
        except Exception as e:
            st.error(f"Error calculating risk score: {str(e)}")
            return None
    # In utils/risk_calculator.py

    def calculate_beta(self, symbol, stock_returns):
        """Calculate beta for a single stock against the market index."""
        try:
            # This requires fetching market index data.
            # A better implementation would have DataFetcher injected or accessible.
            # For now, we'll fetch it here.
            import yfinance as yf
            market_data = yf.download(self.market_index, start=stock_returns.index.min(), end=stock_returns.index.max())
            if market_data.empty:
                logger.warning(f"Could not fetch market data for beta calculation of {symbol}")
                return None

            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align data and calculate covariance
            df = pd.concat([stock_returns, market_returns], axis=1).dropna()
            df.columns = ['Stock', 'Market']
            
            if len(df) < 30: # Need sufficient data points
                return None

            covariance = df.cov().iloc[0, 1]
            market_variance = df['Market'].var()
            
            beta = covariance / market_variance if market_variance != 0 else 1.0
            return beta
        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
            return None

    def calculate_portfolio_beta(self, weights_dict):
        """Calculate the beta of a portfolio given the weights."""
        try:
            portfolio_beta = 0.0
            # A more accurate beta calculation would require fetching individual stock returns.
            # This is a simplified weighted average assuming we have individual betas.
            # For now, we'll return a weighted average of placeholder betas.
            # A real implementation needs individual stock beta calculations.
            
            # This is a simplification. A proper calculation requires individual stock betas.
            # We'll return an approximation.
            # For now, let's assume an average beta of 1.1 for all stocks for the optimizer to work.
            
            placeholder_betas = {symbol: 1.1 for symbol in weights_dict.keys()}

            for symbol, weight in weights_dict.items():
                stock_beta = placeholder_betas.get(symbol, 1.0) # Default to 1 if not found
                portfolio_beta += weight * stock_beta
            
            return portfolio_beta
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0 # Return market beta on error