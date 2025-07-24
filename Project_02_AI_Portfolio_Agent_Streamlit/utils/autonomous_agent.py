# utils/autonomous_agent.py

import os
import streamlit as st
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import utility classes for the Indian market
from .data_fetcher import DataFetcher
from .portfolio_optimizer import PortfolioOptimizer
from .risk_calculator import RiskCalculator
from .database_manager import DatabaseManager
from .intelligent_market_screener import IntelligentMarketScreener
from .ai_analyzer import AIAnalyzer
from .fallback_data_provider import FallbackDataProvider

# Initialize logger
logging.basicConfig(
    filename='portfolio_app_india.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousPortfolioAgent:
    def __init__(self):
        # Initialize AI client using AIAnalyzer
        self.ai_analyzer = AIAnalyzer()
        if not self.ai_analyzer.model:
            st.error("AI Model (Gemini) could not be initialized. AI features will be unavailable.")
            self.client = None
        else:
            self.client = self.ai_analyzer.model
            self.model = "gemini-2.5-pro"  # As defined in AIAnalyzer

        # Initialize components for Indian market
        self.data_fetcher = DataFetcher()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_calculator = RiskCalculator(symbols=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'DABUR.NS', 'UPL.NS'], market_index='^NSEI')
        self.db = DatabaseManager()
        self.market_screener = IntelligentMarketScreener(
            
            finnhub_key=os.getenv('FINNHUB_API_KEY'),
            gemini_key=os.getenv('GEMINI_API_KEY')
        )
        self.fallback_data = FallbackDataProvider()

        # Define Indian stock universe
        self.indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'DABUR.NS', 'UPL.NS']
        self.indian_sectors = {
            'Nifty IT': '^CNXIT',
            'Nifty Bank': '^NSEBANK',
            'Nifty FMCG': '^CNXFMCG',
            'Nifty Auto': '^CNXAUTO',
            'Nifty Pharma': '^CNXPHARMA'
        }

        # Agent persona for Indian market
        self.agent_persona = """
        You are Rakesh Sharma, a legendary portfolio manager with 40 years of experience on Dalal Street.
        
        Experience:
        - Started career in the early 1980s, navigating India's economic liberalization.
        - Managed portfolios through the Harshad Mehta scam, 2000 IT boom/bust, 2008 global financial crisis, demonetization, and GST rollout.
        - Specialized in Indian equities, with expertise in identifying multi-baggers in mid-cap and small-cap segments.
        - Known for value investing and sector rotation strategies tailored to Indian market dynamics.
        - Average annual return of 15.2% over 40-year career.
        
        Investment Philosophy:
        - "In India, conviction in fundamentals and patience outperform market timing."
        - Focus on risk-adjusted returns, leveraging India's growth potential.
        - Diversification across sectors to mitigate policy and monsoon-related volatility.
        - Emphasis on macroeconomic factors like RBI policy, inflation, and government reforms.
        
        Decision Making Style:
        - Data-driven, incorporating Indian market sentiment and policy changes.
        - Long-term perspective with tactical adjustments for sector rotation.
        - Prioritizes risk management, especially against regulatory and FII flow risks.
        - Clear, decisive recommendations with India-centric reasoning.
        """

    def conduct_market_research(self, sectors=['equity']):
        """Conduct comprehensive market research for the Indian equity market."""
        if not self.client:
            return "AI agent not available. Please check GEMINI_API_KEY."

        try:
            # Fetch real-time market data
            market_data = self.data_fetcher.get_batch_stock_data(list(self.indian_sectors.values()) + self.indian_stocks)
            if market_data.empty:
                logger.warning("No real-time data available. Using fallback data for testing.")
                market_data = self.fallback_data.get_synthetic_stock_data(self.indian_stocks + list(self.indian_sectors.values()))
                market_data['source'] = 'synthetic'

            market_summary = self._get_market_summary(market_data)

            prompt = f"""
            {self.agent_persona}
            
            As Rakesh Sharma, provide a comprehensive research report on the Indian equity market.
            
            Current Market Context:
            {market_summary}
            
            Provide a detailed analysis covering:
            1. TOP INVESTMENT THEMES (e.g., Digital India, Make in India, Green Energy).
            2. SECTOR ANALYSIS for Nifty IT, Nifty Bank, Nifty FMCG, Nifty Auto, Nifty Pharma.
            3. RISK MANAGEMENT STRATEGY addressing Indian-specific risks (e.g., inflation, FII outflows, monsoon impact).
            4. MARKET OUTLOOK for Nifty 50 and Sensex (3-6 month perspective).
            
            Cite specific Indian economic indicators (e.g., RBI Repo Rate, CPI Inflation) and policies (e.g., PLI schemes).
            Ensure all monetary values are in INR.
            """
            
            response = self.client.generate_content(prompt, generation_config={'temperature': 0.3, 'max_output_tokens': 5000})
            analysis = response.text.strip()

            # Store analysis in database
            self.db.store_market_analysis(timestamp=datetime.now(), analysis=analysis, source='AIAnalyzer')
            logger.info("Market research stored in database successfully")
            return analysis
        
        except Exception as e:
            logger.error(f"Error conducting market research: {str(e)}")
            st.error(f"Error conducting market research: {str(e)}")
            return "Market research temporarily unavailable"

    def make_autonomous_decisions(self, current_portfolio, market_conditions, risk_tolerance='moderate'):
        """Make autonomous buy/sell decisions for an Indian equity portfolio."""
        if not self.client:
            return "AI agent not available. Please check GEMINI_API_KEY."

        try:
            portfolio_summary = self._analyze_current_portfolio(current_portfolio)
            market_data = self.data_fetcher.get_batch_stock_data(self.indian_stocks + list(self.indian_sectors.values()))
            if market_data.empty:
                logger.warning("No real-time data available. Using fallback data for testing.")
                market_data = self.fallback_data.get_synthetic_stock_data(self.indian_stocks + list(self.indian_sectors.values()))
                market_data['source'] = 'synthetic'

            market_summary = self._get_market_summary(market_data)

            prompt = f"""
            {self.agent_persona}
            
            As Rakesh Sharma, make immediate portfolio decisions for an Indian equity portfolio.
            
            CURRENT PORTFOLIO:
            {portfolio_summary}
            
            CURRENT INDIAN MARKET CONDITIONS:
            {market_summary}
            
            RISK TOLERANCE: {risk_tolerance}
            
            Provide SPECIFIC, ACTIONABLE DECISIONS in INR:
            1. IMMEDIATE ACTIONS (next 24-48 hours):
               - BUY orders: Stock Ticker (e.g., 'RELIANCE.NS'), Quantity, Target Price (₹), Rationale.
               - SELL orders: Stock Ticker, Quantity, Target Price (₹), Rationale.
               - HOLD decisions with rationale.
            2. RISK MANAGEMENT:
               - Stop-loss recommendations for key holdings (in INR).
               - Position sizing adjustments.
            3. REBALANCING RECOMMENDATIONS:
               - Sector allocation adjustments (e.g., "Increase Nifty Bank exposure to 25%").
            4. MARKET TIMING INSIGHTS:
               - Entry/exit points for new positions.
               - Expected market direction (1-3 months).
            
            Use INR for all monetary values and base decisions on Indian market dynamics.
            """
            
            response = self.client.generate_content(prompt, generation_config={'temperature': 0.2, 'max_output_tokens': 5000})
            decisions = response.text.strip()

            # Store decisions in database
            self.db.store_portfolio_recommendation(
                user_session=st.session_state.get('session_id', 'default_session'),
                investment_amount=sum(data.get('market_value', 0) for data in current_portfolio.values()),
                risk_profile=risk_tolerance,
                recommended_allocation={'decisions': decisions},
                analysis=decisions
            )
            logger.info("Autonomous decisions stored in database successfully")
            return decisions
        
        except Exception as e:
            logger.error(f"Error making autonomous decisions: {str(e)}")
            st.error(f"Error making autonomous decisions: {str(e)}")
            return "Decision making temporarily unavailable"

    def predict_market_dynamics(self, time_horizon='3_months'):
        """Predict future dynamics of the Indian equity market."""
        if not self.client:
            return "AI agent not available. Please check GEMINI_API_KEY."

        try:
            market_data = self.data_fetcher.get_batch_stock_data(self.indian_stocks + list(self.indian_sectors.values()))
            if market_data.empty:
                logger.warning("No real-time data available. Using fallback data for testing.")
                market_data = self.fallback_data.get_synthetic_stock_data(self.indian_stocks + list(self.indian_sectors.values()))
                market_data['source'] = 'synthetic'

            current_data = self._gather_predictive_data(market_data)
            horizon_map = {'1_month': '1 month', '3_months': '3 months', '6_months': '6 months', '1_year': '1 year'}

            prompt = f"""
            {self.agent_persona}
            
            As Rakesh Sharma, predict Indian equity market dynamics for the next {horizon_map.get(time_horizon, '3 months')}.
            
            CURRENT INDIAN MARKET DATA:
            {current_data}
            
            Provide a detailed forecast:
            1. NIFTY 50 & SENSEX FORECAST:
               - Direction (Bull/Bear/Sideways) with confidence %.
               - Key support/resistance levels (in INR).
            2. SECTOR ROTATION PREDICTIONS:
               - Outperformers/underperformers among Nifty IT, Nifty Bank, Nifty FMCG, Nifty Auto, Nifty Pharma.
               - Emerging investment themes.
            3. MACROECONOMIC DRIVERS:
               - Impact of RBI policy, CPI inflation, monsoon, and government reforms.
            4. RISK SCENARIOS:
               - Bull case, bear case, and most likely scenario with probabilities.
            
            Base predictions on Indian market cycles and economic indicators.
            """
            
            response = self.client.generate_content(prompt, generation_config={'temperature': 0.5, 'max_output_tokens': 5000})
            prediction = response.text.strip()

            # Store prediction in database
            self.db.store_market_analysis(timestamp=datetime.now(), analysis=prediction, source='AIAnalyzer')
            logger.info("Market prediction stored in database successfully")
            return prediction
        
        except Exception as e:
            logger.error(f"Error predicting market dynamics: {str(e)}")
            st.error(f"Error predicting market dynamics: {str(e)}")
            return "Market prediction temporarily unavailable"

    def create_optimal_portfolio(self, capital_amount, risk_profile='moderate', sectors=['equity']):
        """Create an optimal Indian equity portfolio from scratch."""
        if not self.client:
            return "AI agent not available. Please check GEMINI_API_KEY."

        try:
            session_id = st.session_state.get('session_id', 'default_session')
            logger.info(f"Starting portfolio creation for ₹{capital_amount:,.2f}, risk: {risk_profile}")

            # Fetch real-time data
            with st.spinner("Screening Indian equity market for top opportunities..."):
                best_stocks = self.market_screener.screen_best_opportunities(max_stocks=30, min_score=6.0)
                sector_allocations = self.market_screener.get_sector_allocation_recommendations(best_stocks)

            if not best_stocks:
                logger.warning("No suitable stocks found by screener. Using fallback data for testing.")
                best_stocks = self.fallback_data.get_synthetic_stock_data(self.indian_stocks)
                best_stocks = [
                    {
                        'ticker': row['symbol'],
                        'company_name': row['symbol'].replace('.NS', ''),
                        'investment_score': 7.0,
                        'investment_thesis': 'Synthetic data for testing purposes.',
                        'sector': 'Unknown',
                        'current_price': row['close_price']
                    } for _, row in best_stocks.iterrows()
                ]
                sector_allocations = {'Nifty IT': 0.25, 'Nifty Bank': 0.25, 'Nifty FMCG': 0.20, 'Nifty Auto': 0.15, 'Nifty Pharma': 0.15}

            # Define risk constraints
            risk_constraints = {
                'conservative': {'equity_max': 60, 'cash_min': 40},
                'moderate': {'equity_max': 80, 'cash_min': 20},
                'aggressive': {'equity_max': 95, 'cash_min': 5}
            }
            current_constraints = risk_constraints.get(risk_profile, risk_constraints['moderate'])

            # Fetch market research
            market_research = self.conduct_market_research(sectors)
            market_data = self._gather_predictive_data(self.data_fetcher.get_batch_stock_data(self.indian_stocks + list(self.indian_sectors.values())))

            prompt = f"""
            {self.agent_persona}
            
            As Rakesh Sharma, create an optimal Indian equity portfolio for an investor.
            
            CLIENT PROFILE:
            - Capital: ₹{capital_amount:,.2f}
            - Risk Profile: {risk_profile.upper()}
            - Sectors: Indian Equities (Nifty IT, Nifty Bank, Nifty FMCG, Nifty Auto, Nifty Pharma)
            
            INTELLIGENT SCREENING RESULTS (Top 15 Indian Stocks):
            {json.dumps([{'ticker': s['ticker'], 'company': s.get('company_name', ''), 'score': s.get('investment_score', 0), 'sector': s.get('sector', ''), 'thesis': s.get('investment_thesis', '')} for s in best_stocks[:15]], indent=2)}
            
            SECTOR ALLOCATION RECOMMENDATIONS:
            {json.dumps(sector_allocations, indent=2)}
            
            MANDATORY RISK CONSTRAINTS:
            - Equity: Max {current_constraints['equity_max']}%
            - Cash: Min {current_constraints['cash_min']}%
            
            MARKET RESEARCH:
            {market_research}
            
            CURRENT MARKET DATA:
            {market_data}
            
            Construct a specific portfolio using screened stocks:
            1. EQUITY ALLOCATION (Max {current_constraints['equity_max']}%):
               - Select 8-12 individual stocks from screening results.
               - Include 10-15% in Nifty 50 ETF ('NIFTYBEES.NS') for diversification.
               - Assign specific percentages (3-8% per stock, based on risk profile).
            2. CASH ALLOCATION (Min {current_constraints['cash_min']}%):
               - Allocate to cash for liquidity.
            3. FORMAT FOR EACH POSITION:
               - Ticker (e.g., 'RELIANCE.NS')
               - Company Name (e.g., 'Reliance Industries Ltd.')
               - Allocation Percentage
               - Amount (₹)
               - Investment Rationale (1 sentence)
            
            Ensure total allocation sums to 100%. Use INR for all monetary values.
            """
            
            response = self.client.generate_content(prompt, generation_config={'temperature': 0.2, 'max_output_tokens': 5000})
            ai_response = response.text.strip()

            # Parse AI response for structured portfolio
            portfolio_positions = self._parse_ai_portfolio_response(ai_response, best_stocks, capital_amount, risk_profile, current_constraints)

            # Store portfolio recommendation
            self.db.store_portfolio_recommendation(
                user_session=session_id,
                investment_amount=capital_amount,
                risk_profile=risk_profile,
                recommended_allocation=portfolio_positions,
                analysis=ai_response
            )
            logger.info(f"Portfolio created with {len(portfolio_positions)} positions and stored in database")

            portfolio_result = {
                'ai_analysis': ai_response,
                'total_investment': float(capital_amount),
                'risk_profile': risk_profile,
                'sectors': sectors,
                'portfolio_allocation': portfolio_positions,
                'stock_screening_results': best_stocks[:15],
                'sector_recommendations': sector_allocations,
                'constraints_applied': current_constraints,
                'market_research': market_research,
                'creation_timestamp': datetime.now().isoformat(),
                'screening_summary': {
                    'total_analyzed': len(best_stocks),
                    'selected_for_portfolio': len(portfolio_positions),
                    'methodology': 'Rakesh Sharma Indian Equity Selection'
                }
            }
            return portfolio_result
        
        except Exception as e:
            logger.error(f"Error creating optimal portfolio: {str(e)}")
            st.error(f"Error creating optimal portfolio: {str(e)}")
            return {
                'ai_analysis': f"Error creating portfolio: {str(e)}",
                'total_investment': float(capital_amount),
                'risk_profile': risk_profile,
                'sectors': sectors,
                'portfolio_allocation': {},
                'stock_screening_results': [],
                'sector_recommendations': {},
                'constraints_applied': current_constraints,
                'market_research': "Error occurred during creation",
                'creation_timestamp': datetime.now().isoformat()
            }

    def _parse_ai_portfolio_response(self, ai_response, best_stocks, capital_amount, risk_profile, constraints):
        """Parse AI response to create structured portfolio data."""
        try:
            portfolio = {}
            equity_max = constraints['equity_max'] / 100.0
            cash_min = constraints['cash_min'] / 100.0

            # Simplified parsing: Use top 8-12 stocks from screener
            num_stocks = 12 if risk_profile == 'aggressive' else 10 if risk_profile == 'moderate' else 8
            selected_stocks = best_stocks[:num_stocks]
            if not selected_stocks:
                logger.warning("No stocks available for parsing. Returning empty portfolio.")
                return {}

            total_score = sum(s['investment_score'] for s in selected_stocks)
            equity_alloc = equity_max * 0.85  # Reserve 15% for Nifty ETF
            for stock in selected_stocks:
                weight = (stock['investment_score'] / total_score) * equity_alloc if total_score > 0 else equity_alloc / len(selected_stocks)
                weight = min(weight, 0.08 if risk_profile == 'conservative' else 0.10)  # Cap individual stock allocation
                portfolio[stock['ticker']] = {
                    'company_name': stock.get('company_name', stock['ticker'].replace('.NS', '')),
                    'weight': round(weight, 4),
                    'amount_inr': round(capital_amount * weight, 2),
                    'rationale': stock.get('investment_thesis', 'Selected based on high screening score.'),
                    'sector': stock.get('sector', 'Unknown'),
                    'investment_score': stock.get('investment_score', 0)
                }

            # Add Nifty ETF
            portfolio['NIFTYBEES.NS'] = {
                'company_name': 'Nifty 50 ETF',
                'weight': round(equity_max * 0.15, 4),
                'amount_inr': round(capital_amount * (equity_max * 0.15), 2),
                'rationale': 'Broad market exposure for diversification.',
                'sector': 'Index',
                'investment_score': 7.0
            }

            # Add cash allocation
            cash_weight = max(cash_min, 1.0 - sum(p['weight'] for p in portfolio.values()))
            portfolio['CASH'] = {
                'company_name': 'Cash Reserve',
                'weight': round(cash_weight, 4),
                'amount_inr': round(capital_amount * cash_weight, 2),
                'rationale': 'Liquidity and risk buffer.',
                'sector': 'Cash',
                'investment_score': 0
            }

            # Ensure total allocation is 100%
            total_weight = sum(p['weight'] for p in portfolio.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Portfolio weights sum to {total_weight:.2%}, adjusting cash.")
                portfolio['CASH']['weight'] += 1.0 - total_weight
                portfolio['CASH']['amount_inr'] = round(capital_amount * portfolio['CASH']['weight'], 2)

            logger.info(f"Parsed portfolio with {len(portfolio)} positions")
            return portfolio
        except Exception as e:
            logger.error(f"Error parsing AI portfolio response: {str(e)}")
            return {}

    def _analyze_current_portfolio(self, portfolio_data):
        """Analyze current portfolio holdings in INR."""
        if not portfolio_data:
            return "No current portfolio holdings"

        try:
            summary = "Current Portfolio Analysis:\n"
            total_value = sum(data.get('market_value', 0) for data in portfolio_data.values())
            summary += f"Total Portfolio Value: ₹{total_value:,.2f}\n\nCurrent Allocation:\n"

            for symbol, data in portfolio_data.items():
                if 'market_value' in data:
                    percentage = (data['market_value'] / total_value) * 100 if total_value > 0 else 0
                    summary += f"- {symbol}: {percentage:.1f}% (₹{data['market_value']:,.2f})\n"

            return summary
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            return "Portfolio analysis unavailable"

    def _get_market_summary(self, market_data):
        """Get a summary of the current Indian equity market."""
        try:
            summary = "Current Indian Equity Market Summary:\n"
            nifty_data = market_data[market_data['symbol'] == '^NSEI']
            sensex_data = market_data[market_data['symbol'] == '^BSESN']

            if not nifty_data.empty:
                summary += f"- Nifty 50: ₹{nifty_data['close_price'].iloc[-1]:,.2f}, Change: {nifty_data['pct_change'].iloc[-1]:.2f}%\n"
            if not sensex_data.empty:
                summary += f"- Sensex: ₹{sensex_data['close_price'].iloc[-1]:,.2f}, Change: {sensex_data['pct_change'].iloc[-1]:.2f}%\n"

            summary += "\nSector Performance:\n"
            for sector_name, ticker in self.indian_sectors.items():
                sector_data = market_data[market_data['symbol'] == ticker]
                if not sector_data.empty:
                    summary += f"- {sector_name}: ₹{sector_data['close_price'].iloc[-1]:,.2f}, Change: {sector_data['pct_change'].iloc[-1]:.2f}%\n"

            summary += "\nKey Themes:\n- Digital India and PLI schemes driving growth.\n- Strong domestic retail participation.\n- FII flows mixed amid global uncertainties.\n"
            summary += "Risks:\n- Inflation pressures, RBI rate decisions, monsoon variability, and geopolitical tensions.\n"
            return summary
        except Exception as e:
            logger.error(f"Error generating market summary: {str(e)}")
            return "Market summary unavailable"

    def _gather_predictive_data(self, market_data):
        """Gather data for Indian equity market predictions."""
        try:
            data_summary = {
                'market_indices': {
                    'Nifty 50': f"₹{market_data[market_data['symbol'] == '^NSEI']['close_price'].iloc[-1]:,.2f}" if not market_data[market_data['symbol'] == '^NSEI'].empty else 'N/A',
                    'Sensex': f"₹{market_data[market_data['symbol'] == '^BSESN']['close_price'].iloc[-1]:,.2f}" if not market_data[market_data['symbol'] == '^BSESN'].empty else 'N/A'
                },
                'economic_indicators': {
                    'RBI Repo Rate': self.db.get_latest_economic_indicator('RBI Repo Rate', 'India') or '6.50%',
                    'CPI Inflation': self.db.get_latest_economic_indicator('CPI Inflation', 'India') or '4.9%',
                    'GDP Growth (YoY)': self.db.get_latest_economic_indicator('GDP Growth', 'India') or '7.8%'
                },
                'key_themes': ['Digital India', 'Make in India', 'Green Energy', 'Financial Inclusion'],
                'risk_factors': ['FII outflows', 'Monsoon variability', 'Global slowdown', 'Regulatory changes']
            }
            return json.dumps(data_summary, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error gathering predictive data: {str(e)}")
            return json.dumps({'error': 'Data unavailable'}, indent=2)