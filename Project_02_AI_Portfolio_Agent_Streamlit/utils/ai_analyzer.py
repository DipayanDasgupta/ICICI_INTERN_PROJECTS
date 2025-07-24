# utils/ai_analyzer.py

import google.generativeai as genai
import os
import streamlit as st
import json
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import logging # FIX: Import the logging module

# FIX: Create a logger instance for this file
logger = logging.getLogger(__name__)

class AIAnalyzer:
    def __init__(self):
        # Initialize the Gemini client
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        try:
            if gemini_key:
                genai.configure(api_key=gemini_key)
                # Using gemini-1.5-pro-latest is a good choice for fast and capable analysis
                self.model = genai.GenerativeModel('gemini-2.5-pro')
            else:
                st.error("GEMINI_API_KEY not found. Please add it to your environment variables.")
                self.model = None
        except Exception as e:
            st.error(f"Error initializing Gemini client: {str(e)}")
            self.model = None
    
    def _fetch_market_data(self, symbols: list, period: str = "1mo") -> dict:
        """Fetch market data for given symbols using yfinance."""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # FIX: Add auto_adjust=False to maintain the old data structure (OHLC) and suppress the warning.
            data = yf.download(symbols + ['^NSEI'], period=period, progress=False, auto_adjust=False)
            
            if data.empty:
                return {}
            
            market_data = {}
            for symbol in symbols:
                if symbol in data['Adj Close'].columns:
                    price_data = data['Adj Close'][symbol].dropna()
                    if not price_data.empty:
                        latest_price = price_data.iloc[-1]
                        returns = price_data.pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                        market_data[symbol] = {
                            'current_price': latest_price,
                            'volatility_annualized': volatility,
                            'momentum_30d': ((price_data.iloc[-1] / price_data.iloc[0]) - 1) * 100 if len(price_data) >= 2 else 0
                        }
            
            # Add Nifty 50 data
            if '^NSEI' in data['Adj Close'].columns:
                nifty_data = data['Adj Close']['^NSEI'].dropna()
                if not nifty_data.empty:
                    market_data['^NSEI'] = {
                        'current_price': nifty_data.iloc[-1],
                        'volatility_annualized': nifty_data.pct_change().dropna().std() * np.sqrt(252) if len(nifty_data) > 1 else 0
                    }
            
            return market_data
        except Exception as e:
            st.error(f"Error fetching market data with yfinance: {str(e)}")
            return {}

    def get_market_analysis(self, market_context, economic_data=None):
        """Get AI-powered market analysis for Indian market"""
        if not self.model:
            return "AI analysis temporarily unavailable"
        
        try:
            # Fetch market data for Nifty 50 to enrich context
            market_data = self._fetch_market_data(['^NSEI'])
            volatility = market_data.get('^NSEI', {}).get('volatility_annualized', 0)
            
            if isinstance(volatility, (int, float)):
                volatility_str = f"{volatility:.2%}"
            else:
                volatility_str = "N/A"

            market_data_str = (f"Nifty 50 Index: Current Level={market_data.get('^NSEI', {}).get('current_price', 'N/A')}, "
                             f"Volatility (Annualized)={volatility_str}")
            
            prompt = f"""
            As an expert financial advisor and portfolio manager specializing in the Indian market, analyze the current market conditions and provide insights.
            
            Current Market Context:
            {market_context}
            
            Economic Data:
            {economic_data if economic_data else 'Economic data not available. Consider factors like GDP growth, inflation, RBI policy rates, and rupee exchange rate.'}
            
            Market Data:
            {market_data_str}
            
            Please provide:
            1. A brief market outlook (2-3 sentences)
            2. Key risks to monitor (e.g., monsoon impacts, regulatory changes, rupee depreciation)
            3. One actionable insight for portfolio management
            
            Keep the response concise and focused on actionable insights for retail investors in India.
            """
            
            response = self.model.generate_content(
                contents=prompt,
                generation_config={
                    'max_output_tokens': 10000,
                    'temperature': 0.3
                },
                request_options={'timeout': 120}
            )

            try:
                response_text = response.text
                logger.debug(f"DEBUG: AI response text: {repr(response_text[:100])}")
                return response_text
            except ValueError:
                logger.warning(f"Gemini response blocked. Finish Reason: {response.candidates[0].finish_reason}. Prompt Feedback: {response.prompt_feedback}")
                return "AI analysis was blocked due to safety settings. Please try rephrasing your request."
        
        except Exception as e:
            logger.error(f"DEBUG: Error in get_market_analysis: {str(e)}")
            st.error(f"Error getting market analysis: {str(e)}")
            return "AI analysis temporarily unavailable"
    
    # ... (the rest of the file remains the same) ...
    # ... (analyze_portfolio, get_rebalancing_recommendations, etc.) ...
    # Make sure to include the rest of your AIAnalyzer class methods here.
    
    def analyze_portfolio(self, portfolio_data, market_data=None):
        """Analyze current portfolio composition and performance for Indian stocks"""
        if not self.model:
            return "AI analysis temporarily unavailable"
        
        try:
            portfolio_summary = self._format_portfolio_data(portfolio_data)
            
            # Fetch market data for portfolio symbols
            symbols = list(portfolio_data.keys())
            fetched_market_data = self._fetch_market_data(symbols)
            market_data_str = market_data if market_data else ""
            if fetched_market_data:
                market_data_str += "\nMarket Data (via yfinance):\n"
                for symbol, data in fetched_market_data.items():
                    if symbol != '^NSEI':
                        market_data_str += f"- {symbol}: Price=₹{data['current_price']:,.2f}, Volatility={data['volatility_annualized']:.2%}, 30d Momentum={data['momentum_30d']:.2%}\n"
            
            prompt = f"""
            As a portfolio management expert for the Indian market, analyze this portfolio composition:
            
            {portfolio_summary}
            
            Market Context:
            {market_data_str if market_data_str else 'Limited market data available. Consider Nifty 50 trends and sector performance.'}
            
            Please provide:
            1. Portfolio diversification assessment
            2. Risk level evaluation (Low/Medium/High)
            3. Top 2 specific recommendations for improvement
            4. One asset class that might be missing (e.g., bonds, gold, Indian small-caps)
            
            Focus on practical, actionable advice for a retail investor in India.
            """
            
            response = self.model.generate_content(
                contents=prompt,
                generation_config={
                    'max_output_tokens': 10000,
                    'temperature': 0.3
                },
                request_options={'timeout': 120} 
            )
            
            return response.text
        
        except Exception as e:
            st.error(f"Error analyzing portfolio: {str(e)}")
            return "Portfolio analysis temporarily unavailable"
    
    def get_rebalancing_recommendations(self, portfolio_data, target_allocation=None):
        """Get AI-powered portfolio rebalancing recommendations for Indian stocks"""
        if not self.model:
            return "AI recommendations temporarily unavailable"
        
        try:
            portfolio_summary = self._format_portfolio_data(portfolio_data)
            
            # Fetch market data for portfolio symbols
            symbols = list(portfolio_data.keys())
            fetched_market_data = self._fetch_market_data(symbols)
            market_data_str = ""
            if fetched_market_data:
                market_data_str = "Market Data (via yfinance):\n"
                for symbol, data in fetched_market_data.items():
                    if symbol != '^NSEI':
                        market_data_str += f"- {symbol}: Price=₹{data['current_price']:,.2f}, Volatility={data['volatility_annualized']:.2%}\n"
            
            prompt = f"""
            As a portfolio rebalancing expert for the Indian market, analyze this portfolio and provide rebalancing recommendations:
            
            Current Portfolio:
            {portfolio_summary}
            
            Target Allocation (if specified):
            {target_allocation if target_allocation else 'No specific target provided - suggest optimal allocation based on Indian market conditions'}
            
            Market Data:
            {market_data_str if market_data_str else 'Limited market data available'}
            
            Please provide:
            1. Current allocation analysis
            2. Recommended target allocation percentages (e.g., across Nifty sectors or asset classes like equities, bonds, gold)
            3. Specific rebalancing actions (buy/sell/hold)
            4. Rationale for each recommendation
            
            Consider Indian market conditions (e.g., RBI policies, monsoon impacts, sector trends) and maintain appropriate diversification.
            """
            
            response = self.model.generate_content(
                contents=prompt,
                generation_config={
                    'max_output_tokens': 5000,
                    'temperature': 0.3
                },
                request_options={'timeout': 120} 
            )
            
            return response.text
        
        except Exception as e:
            st.error(f"Error getting rebalancing recommendations: {str(e)}")
            return "Rebalancing recommendations temporarily unavailable"
    
    def analyze_news_sentiment(self, news_data):
        """Analyze sentiment of market news and provide insights for Indian market"""
        if not self.model or not news_data:
            return "News analysis not available"
        
        try:
            news_summary = ""
            for item in news_data[:5]:  # Analyze top 5 news items
                news_summary += f"- {item['title']}: {item['summary'][:100]}...\n"
            
            prompt = f"""
            Analyze the sentiment and implications of these recent market news items for the Indian market:
            
            {news_summary}
            
            Please provide:
            1. Overall market sentiment (Bullish/Bearish/Neutral)
            2. Key themes or trends identified (e.g., policy changes, sector performance)
            3. Potential impact on different Indian asset classes (e.g., equities, bonds, gold)
            4. One actionable insight for portfolio positioning in India
            
            Keep the analysis concise and focused on investment implications for Indian investors.
            """
            
            response = self.model.generate_content(
                contents=prompt,
                generation_config={
                    'max_output_tokens': 5000,
                    'temperature': 0.3
                },
                request_options={'timeout': 120} 
            )
            
            return response.text
        
        except Exception as e:
            st.error(f"Error analyzing news sentiment: {str(e)}")
            return "News sentiment analysis temporarily unavailable"
    
    def get_risk_assessment(self, portfolio_data, volatility_data=None, correlation_data=None):
        """Get AI-powered risk assessment of the portfolio for Indian market"""
        if not self.model:
            return "Risk assessment temporarily unavailable"
        
        try:
            portfolio_summary = self._format_portfolio_data(portfolio_data)
            
            # Fetch market data for portfolio symbols
            symbols = list(portfolio_data.keys())
            fetched_market_data = self._fetch_market_data(symbols)
            volatility_data_str = volatility_data if volatility_data else ""
            if fetched_market_data:
                volatility_data_str += "\nVolatility Data (via yfinance):\n"
                for symbol, data in fetched_market_data.items():
                    if symbol != '^NSEI':
                        volatility_data_str += f"- {symbol}: Annualized Volatility={data['volatility_annualized']:.2%}\n"
            
            prompt = f"""
            As a risk management expert for the Indian market, assess the risk profile of this portfolio:
            
            Portfolio Composition:
            {portfolio_summary}
            
            Volatility Data:
            {volatility_data_str if volatility_data_str else 'Volatility data not available'}
            
            Correlation Data:
            {correlation_data if correlation_data else 'Correlation data not available'}
            
            Please provide:
            1. Overall risk level (Conservative/Moderate/Aggressive)
            2. Main risk factors identified (e.g., sector concentration, rupee volatility)
            3. Diversification effectiveness
            4. Risk mitigation recommendations (e.g., hedging with gold, sector rebalancing)
            
            Focus on practical risk management advice for Indian retail investors.
            """
            
            response = self.model.generate_content(
                contents=prompt,
                generation_config={
                    'max_output_tokens': 5000,
                    'temperature': 0.3
                },
                request_options={'timeout': 120} 
            )
            
            return response.text
        
        except Exception as e:
            st.error(f"Error getting risk assessment: {str(e)}")
            return "Risk assessment temporarily unavailable"
    
    def generate_investment_thesis(self, symbol, market_data=None, economic_context=None):
        """Generate investment thesis for a specific Indian stock"""
        if not self.model:
            return "Investment thesis generation temporarily unavailable"
        
        try:
            # Fetch market data for the symbol
            fetched_market_data = self._fetch_market_data([symbol])
            market_data_str = market_data if market_data else ""
            if fetched_market_data and symbol in fetched_market_data:
                data = fetched_market_data[symbol]
                market_data_str += f"\nMarket Data for {symbol} (via yfinance):\n" \
                                  f"- Current Price: ₹{data['current_price']:,.2f}\n" \
                                  f"- Volatility (Annualized): {data['volatility_annualized']:.2%}\n" \
                                  f"- 30-day Momentum: {data['momentum_30d']:.2%}\n"
            
            # Fetch stock info for sector and other details
            stock = yf.Ticker(symbol)
            info = stock.info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            prompt = f"""
            As an investment analyst specializing in the Indian market, create a concise investment thesis for {symbol} ({industry}, {sector}):
            
            Market Data Context:
            {market_data_str if market_data_str else 'Limited market data available'}
            
            Economic Context:
            {economic_context if economic_context else 'Economic context not provided. Consider RBI policies, monsoon impacts, and rupee trends.'}
            
            Please provide:
            1. Bull case (2-3 key points)
            2. Bear case (2-3 key points)
            3. Fair value assessment
            4. Recommended position size (% of portfolio)
            
            Keep the analysis balanced and practical for retail investors in India.
            """
            
            response = self.model.generate_content(
                contents=prompt,
                generation_config={
                    'max_output_tokens': 5000,
                    'temperature': 0.3
                },
                request_options={'timeout': 120} 
            )
            
            return response.text
        
        except Exception as e:
            st.error(f"Error generating investment thesis: {str(e)}")
            return "Investment thesis generation temporarily unavailable"
    
    def _format_portfolio_data(self, portfolio_data):
        """Format portfolio data for AI analysis in INR"""
        if not portfolio_data:
            return "No portfolio data available"
        
        summary = "Portfolio Holdings:\n"
        total_value = 0
        
        for symbol, data in portfolio_data.items():
            if isinstance(data, dict) and 'market_value' in data:
                summary += f"- {symbol}: ₹{data['market_value']:,.2f} ({data['shares']} shares at ₹{data['current_price']:,.2f})\n"
                total_value += data['market_value']
            else:
                summary += f"- {symbol}: Holdings data available\n"
        
        if total_value > 0:
            summary += f"\nTotal Portfolio Value: ₹{total_value:,.2f}\n"
            summary += "\nAllocation Percentages:\n"
            for symbol, data in portfolio_data.items():
                if isinstance(data, dict) and 'market_value' in data:
                    percentage = (data['market_value'] / total_value) * 100
                    summary += f"- {symbol}: {percentage:.1f}%\n"
        
        return summary