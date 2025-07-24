import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st
from datetime import datetime, timedelta
import json
import logging
from .data_fetcher import DataFetcher
from .database_manager import DatabaseManager
from .risk_calculator import RiskCalculator

class PortfolioOptimizer:
    def __init__(self, symbols=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']):
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = 0.06  # 6% annual risk-free rate (approx. Indian 10-year G-Sec yield, 2025)
        self.data_fetcher = DataFetcher()
        self.db_manager = DatabaseManager()
        self.risk_calculator = RiskCalculator(symbols, market_index='^NSEI')
        self.symbols = symbols
    
    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio return, volatility, Sharpe ratio, and beta for Indian stocks"""
        try:
            # Annualized portfolio return
            portfolio_return = np.sum(returns.mean() * weights) * 252  # 252 trading days
            
            # Portfolio volatility (annualized)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            
            # Portfolio beta using RiskCalculator
            weights_dict = {symbol: weight for symbol, weight in zip(returns.columns, weights)}
            portfolio_beta = self.risk_calculator.calculate_portfolio_beta(weights_dict)
            if portfolio_beta is None:
                portfolio_beta = 1.0  # Default to market beta if calculation fails
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'beta': portfolio_beta
            }
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return None
    
    def optimize_portfolio(self, returns_data, optimization_type='sharpe'):
        """
        Optimize portfolio allocation for Indian stocks
        optimization_type: 'sharpe', 'min_vol', 'max_return', 'min_beta'
        """
        try:
            if returns_data.empty:
                self.logger.warning("Empty returns data provided for optimization")
                return None
            
            # Calculate returns and covariance matrix
            returns = returns_data.pct_change().dropna()
            if returns.empty:
                self.logger.warning("No valid returns calculated from data")
                return None
                
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(returns.columns)
            
            # Constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
            bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
            
            # Initial guess (equal weights)
            initial_guess = np.array([1/num_assets] * num_assets)
            
            # Objective functions
            def negative_sharpe_ratio(weights):
                metrics = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
                return -metrics['sharpe_ratio'] if metrics else float('inf')
            
            def portfolio_volatility(weights):
                metrics = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
                return metrics['volatility'] if metrics else float('inf')
            
            def negative_return(weights):
                metrics = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
                return -metrics['return'] if metrics else float('inf')
            
            def portfolio_beta(weights):
                metrics = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
                return metrics['beta'] if metrics else float('inf')
            
            # Choose objective function
            objective_map = {
                'sharpe': negative_sharpe_ratio,
                'min_vol': portfolio_volatility,
                'max_return': negative_return,
                'min_beta': portfolio_beta
            }
            objective = objective_map.get(optimization_type, negative_sharpe_ratio)
            
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                metrics = self.calculate_portfolio_metrics(optimal_weights, returns, cov_matrix)
                
                # Store optimization result in DatabaseManager
                if self.db_manager.is_available():
                    allocation = {symbol: weight for symbol, weight in zip(returns.columns, optimal_weights)}
                    self.db_manager.store_portfolio_recommendation(
                        user_session=f"optimization_{optimization_type}_{datetime.now().isoformat()}",
                        investment_amount=1000000,  # Default amount in INR
                        risk_profile=optimization_type.upper(),
                        recommended_allocation=allocation,
                        ai_analysis=f"Optimized portfolio: Return {metrics['return']:.2%}, Volatility {metrics['volatility']:.2%}, Sharpe {metrics['sharpe_ratio']:.2f}, Beta {metrics['beta']:.2f}"
                    )
                
                return {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'metrics': metrics,
                    'success': True
                }
            else:
                self.logger.warning(f"Optimization failed: {result.message}")
                return {'success': False, 'message': result.message}
        
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            st.error(f"Error in portfolio optimization: {str(e)}")
            return None
    
    def generate_efficient_frontier(self, returns_data, num_portfolios=100):
        """Generate efficient frontier data for Indian stocks"""
        try:
            if returns_data.empty:
                self.logger.warning("Empty returns data provided for efficient frontier")
                return None
            
            returns = returns_data.pct_change().dropna()
            if returns.empty:
                self.logger.warning("No valid returns calculated for efficient frontier")
                return None
                
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(returns.columns)
            
            # Target returns for efficient frontier
            min_ret = mean_returns.min() * 252
            max_ret = mean_returns.max() * 252
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                # Constraints
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                    {'type': 'eq', 'fun': lambda x, target=target_return: 
                     np.sum(mean_returns * x) * 252 - target}  # Target return
                ]
                
                bounds = tuple((0, 1) for _ in range(num_assets))
                initial_guess = np.array([1/num_assets] * num_assets)
                
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                
                result = minimize(
                    portfolio_volatility,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    weights = result.x
                    volatility = portfolio_volatility(weights)
                    sharpe = (target_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                    beta = self.calculate_portfolio_metrics(weights, returns, cov_matrix).get('beta', 1.0)
                    
                    portfolio_info = {
                        'return': target_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe,
                        'beta': beta,
                        'weights': dict(zip(returns.columns, weights))
                    }
                    efficient_portfolios.append(portfolio_info)
                    
                    # Store in DatabaseManager
                    if self.db_manager.is_available():
                        self.db_manager.store_portfolio_recommendation(
                            user_session=f"efficient_frontier_{target_return:.4f}_{datetime.now().isoformat()}",
                            investment_amount=1000000,  # Default amount in INR
                            risk_profile="EFFICIENT_FRONTIER",
                            recommended_allocation=portfolio_info['weights'],
                            ai_analysis=f"Efficient frontier point: Return {target_return:.2%}, Volatility {volatility:.2%}, Sharpe {sharpe:.2f}, Beta {beta:.2f}"
                        )
            
            return efficient_portfolios
        
        except Exception as e:
            self.logger.error(f"Error generating efficient frontier: {str(e)}")
            st.error(f"Error generating efficient frontier: {str(e)}")
            return None
    
    def calculate_current_allocation(self, portfolio_data):
        """Calculate current portfolio allocation percentages"""
        try:
            total_value = sum([holding.get('market_value', 0) for holding in portfolio_data.values()])
            
            if total_value == 0:
                self.logger.warning("Total portfolio value is zero")
                return {}
            
            allocation = {}
            for symbol, data in portfolio_data.items():
                market_value = data.get('market_value', 0)
                allocation[symbol] = market_value / total_value
            
            return allocation
        
        except Exception as e:
            self.logger.error(f"Error calculating current allocation: {str(e)}")
            st.error(f"Error calculating current allocation: {str(e)}")
            return {}
    
    def suggest_rebalancing(self, current_portfolio, target_allocation, total_value):
        """Suggest specific rebalancing actions for Indian portfolio in INR"""
        try:
            current_allocation = self.calculate_current_allocation(current_portfolio)
            suggestions = []
            
            for symbol in set(list(current_allocation.keys()) + list(target_allocation.keys())):
                current_weight = current_allocation.get(symbol, 0)
                target_weight = target_allocation.get(symbol, 0)
                
                difference = target_weight - current_weight
                amount_difference = difference * total_value  # INR
                
                if abs(amount_difference) > 1000:  # Minimum threshold in INR
                    action = "BUY" if difference > 0 else "SELL"
                    amount_difference = abs(amount_difference)
                    
                    suggestions.append({
                        'symbol': symbol,
                        'action': action,
                        'amount': amount_difference,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_difference': difference
                    })
            
            # Sort by largest absolute difference
            suggestions.sort(key=lambda x: abs(x['amount']), reverse=True)
            
            # Store rebalancing suggestions in DatabaseManager
            if self.db_manager.is_available() and suggestions:
                self.db_manager.store_portfolio_recommendation(
                    user_session=f"rebalancing_{datetime.now().isoformat()}",
                    investment_amount=total_value,
                    risk_profile="REBALANCING",
                    recommended_allocation=target_allocation,
                    ai_analysis=json.dumps(suggestions)
                )
            
            return suggestions
        
        except Exception as e:
            self.logger.error(f"Error generating rebalancing suggestions: {str(e)}")
            st.error(f"Error generating rebalancing suggestions: {str(e)}")
            return []
    
    def monte_carlo_simulation(self, returns_data, weights, num_simulations=1000, time_horizon=252):
        """Run Monte Carlo simulation for Indian portfolio performance"""
        try:
            if returns_data.empty:
                self.logger.warning("Empty returns data provided for Monte Carlo simulation")
                return None
            
            returns = returns_data.pct_change().dropna()
            if returns.empty:
                self.logger.warning("No valid returns calculated for Monte Carlo simulation")
                return None
                
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Portfolio statistics
            weights_array = np.array([weights.get(symbol, 0) for symbol in returns.columns])
            portfolio_mean = np.sum(mean_returns * weights_array)
            portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
            
            # Monte Carlo simulation
            simulated_returns = np.random.normal(
                portfolio_mean, 
                portfolio_std, 
                (num_simulations, time_horizon)
            )
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
            final_values = cumulative_returns[:, -1]
            
            # Calculate statistics
            percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
            
            simulation_result = {
                'simulations': cumulative_returns,
                'final_values': final_values,
                'percentiles': {
                    '5th': percentiles[0],
                    '25th': percentiles[1],
                    'median': percentiles[2],
                    '75th': percentiles[3],
                    '95th': percentiles[4]
                },
                'mean_final_value': np.mean(final_values),
                'std_final_value': np.std(final_values)
            }
            
            # Store simulation results in DatabaseManager
            if self.db_manager.is_available():
                self.db_manager.store_portfolio_recommendation(
                    user_session=f"monte_carlo_{datetime.now().isoformat()}",
                    investment_amount=1000000,  # Default amount in INR
                    risk_profile="MONTE_CARLO",
                    recommended_allocation=weights,
                    ai_analysis=f"Monte Carlo simulation: Median return {simulation_result['percentiles']['median']:.2%}, "
                               f"5th percentile {simulation_result['percentiles']['5th']:.2%}, "
                               f"95th percentile {simulation_result['percentiles']['95th']:.2%}"
                )
            
            return simulation_result
        
        except Exception as e:
            self.logger.error(f"Error running Monte Carlo simulation: {str(e)}")
            st.error(f"Error running Monte Carlo simulation: {str(e)}")
            return None
    
    def calculate_drawdown(self, returns_series):
        """Calculate maximum drawdown for a returns series"""
        try:
            cumulative = (1 + returns_series).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Integrate with RiskCalculator for consistency
            risk_drawdown = self.risk_calculator.calculate_max_drawdown(returns_series.sum(), returns_series.sum())
            
            drawdown_result = {
                'max_drawdown': min(max_drawdown, risk_drawdown) if risk_drawdown is not None else max_drawdown,
                'drawdown_series': drawdown
            }
            
            # Store drawdown in DatabaseManager
            if self.db_manager.is_available():
                self.db_manager.store_portfolio_recommendation(
                    user_session=f"drawdown_{datetime.now().isoformat()}",
                    investment_amount=1000000,  # Default amount in INR
                    risk_profile="DRAWDOWN",
                    recommended_allocation={},
                    ai_analysis=f"Max drawdown: {drawdown_result['max_drawdown']:.2%}"
                )
            
            return drawdown_result
        
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {str(e)}")
            st.error(f"Error calculating drawdown: {str(e)}")
            return None