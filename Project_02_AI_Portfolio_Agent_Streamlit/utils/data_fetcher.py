# utils/data_fetcher.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import streamlit as st
import yfinance as yf
import finnhub
import google.generativeai as genai
from .database_manager import DatabaseManager
import logging
from .fallback_data_provider import FallbackDataProvider # --- FIX: Import FallbackDataProvider

class DataFetcher:
    def __init__(self):
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        
        self.cache_duration = 300
        self.db = DatabaseManager()
        self.fallback = FallbackDataProvider() # --- FIX: Initialize FallbackDataProvider
         
        try:
            self.finnhub_client = finnhub.Client(api_key=self.finnhub_key) if self.finnhub_key else None
        except Exception as e:
            st.error(f"Error initializing Finnhub client: {str(e)}")
            self.finnhub_client = None

        try:
            if self.gemini_key:
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            else:
                self.gemini_model = None
        except Exception as e:
            st.error(f"Error initializing Gemini client: {str(e)}")
            self.gemini_model = None

        self.logger = logging.getLogger(__name__) # --- FIX: Use correct logger name
        self.logger.info("Initialized DataFetcher for Indian stocks with yfinance, Finnhub, and Gemini")
        self.nifty_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'DABUR.NS', 'UPL.NS']

    def get_stock_data(self, symbol, period='1y'):
        """
        Fetch stock data robustly, prioritizing yfinance and using fallbacks correctly.
        """
        try:
            # --- START OF FIX 1: Correct argument for data_freshness_check ---
            # Use `max_age_seconds` instead of `max_age_hours`
            if self.db.data_freshness_check(symbol, 'stock', max_age_seconds=3600): # 1 hour cache
                cached_data = self.db.get_market_data(symbol, 'stock', days_back=365)
                if cached_data is not None and not cached_data.empty:
                    self.logger.info(f"Using CACHED data for {symbol}")
                    cached_data['timestamp'] = pd.to_datetime(cached_data['timestamp'], errors='coerce')
                    cached_data.set_index('timestamp', inplace=True)
                    cached_data.index = cached_data.index.tz_localize(None) # Standardize timezone
                    df = cached_data[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    return df.astype(float)
            # --- END OF FIX 1 ---

            # 2. If no fresh cache, fetch from yfinance (primary source)
            self.logger.info(f"Cache miss or stale. Fetching LIVE data for {symbol} from yfinance.")
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval="1d", auto_adjust=False)
            
            if df.empty:
                self.logger.warning(f"No yfinance data found for {symbol}. Trying Finnhub.")
                df_finnhub = self._get_stock_data_finnhub(symbol)
                if df_finnhub is not None and not df_finnhub.empty:
                    return df_finnhub # Return Finnhub data if successful
                else:
                    self.logger.warning(f"Finnhub also failed for {symbol}. Using SYNTHETIC data.")
                    return self.fallback.generate_stock_data(symbol, days=252) # Use final fallback

            # --- START OF FIX 2: Standardize timezone for fresh data ---
            df.index = df.index.tz_localize(None)
            # --- END OF FIX 2 ---

            # 3. Store newly fetched yfinance data
            df_for_db = df.copy()
            df_for_db.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            df_for_db['timestamp'] = df_for_db.index
            self.db.store_market_data(symbol, 'stock', df_for_db)
            
            self.logger.info(f"Successfully fetched and stored {symbol} from yfinance")
            return df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

        except Exception as e:
            self.logger.error(f"Critical error in get_stock_data for {symbol}: {e}. Returning synthetic data as last resort.")
            return self.fallback.generate_stock_data(symbol, days=252)

    # --- (the rest of the file can remain the same, as the primary logic above is now robust) ---
    def _get_stock_data_finnhub(self, symbol):
        """Fetch stock data from Finnhub API using free endpoints"""
        try:
            if not self.finnhub_client:
                self.logger.error("Finnhub client not initialized")
                return None

            end_date = int(datetime.now().timestamp())
            start_date = int((datetime.now() - timedelta(days=365)).timestamp())
            candles = self.finnhub_client.stock_candles(symbol, 'D', start_date, end_date)

            if candles.get('s') != 'ok' or not candles.get('c'):
                self.logger.warning(f"No historical data from Finnhub for {symbol}")
                return None

            df_data = {'open': candles['o'], 'high': candles['h'], 'low': candles['l'], 'close': candles['c'], 'volume': candles['v']}
            dates = pd.to_datetime(candles['t'], unit='s')
            df = pd.DataFrame(df_data, index=dates).sort_index()

            df_for_db = df.copy()
            df_for_db['timestamp'] = df_for_db.index
            self.db.store_market_data(symbol, 'stock', df_for_db)
            self.logger.info(f"Successfully fetched {symbol} from Finnhub")
            return df
        except Exception as e:
            self.logger.error(f"Finnhub API error for {symbol}: {str(e)}")
            return None
 

    def get_economic_indicators(self):
        """Fetch macroeconomic indicators for India using Finnhub"""
        indicators = {}
        try:
            if not self.finnhub_client:
                self.logger.error("Finnhub client not initialized")
                return indicators

            # Fetch India-specific economic indicators via Finnhub economic calendar
            try:
                economic_data = self.finnhub_client.economic_calendar()
                if economic_data and 'economicCalendar' in economic_data:
                    for event in economic_data['economicCalendar']:
                        if 'India' in event.get('country', '') and 'actual' in event:
                            date = event.get('eventTime', datetime.now().strftime('%Y-%m-%d'))
                            if 'CPI' in event['event']:
                                indicators['india_cpi'] = {'value': float(event['actual']), 'date': date}
                            elif 'GDP' in event['event']:
                                indicators['india_gdp_growth'] = {'value': float(event['actual']), 'date': date}
                            elif 'Interest Rate' in event['event'] or 'Repo Rate' in event['event']:
                                indicators['rbi_repo_rate'] = {'value': float(event['actual']), 'date': date}
            except Exception as e:
                self.logger.warning(f"Could not fetch economic indicators from Finnhub: {str(e)}")

            if not indicators:
                self.logger.warning("No economic indicators available from Finnhub for India")
            else:
                self.logger.info("Fetched economic indicators for India from Finnhub")
            return indicators

        except Exception as e:
            st.warning(f"Could not fetch economic indicators: {str(e)}")
            return indicators

    @st.cache_data(ttl=1800)
    def get_market_news(self):
        """Fetch and analyze market news for Indian stocks using Gemini"""
        if not self.gemini_model:
            self.logger.error("Gemini model not initialized")
            return []

        try:
            # Fetch recent news from Finnhub for representative Nifty stocks
            news_items = []
            if self.finnhub_client:
                for symbol in self.nifty_tickers[:5]:  # Limit to avoid rate limits
                    try:
                        news = self.finnhub_client.company_news(
                            symbol,
                            _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                            to=datetime.now().strftime('%Y-%m-%d')
                        )
                        for item in news[:2]:  # Take top 2 news per symbol
                            news_items.append({
                                'title': item.get('headline', 'No title'),
                                'summary': item.get('summary', 'No summary'),
                                'url': item.get('url', ''),
                                'time_published': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                                'source': item.get('source', 'Unknown')
                            })
                    except Exception as e:
                        self.logger.warning(f"Could not fetch news for {symbol} from Finnhub: {str(e)}")

            if not news_items:
                self.logger.warning("No news data available from Finnhub")
                return []

            # Analyze news sentiment with Gemini
            news_summary = ""
            for item in news_items[:5]:  # Limit to top 5 news items
                news_summary += f"- {item['title']}: {item['summary'][:100]}...\n"

            prompt = f"""
            Analyze the sentiment and implications of these recent market news items for the Indian market:
            
            {news_summary}
            
            Please provide:
            1. Overall market sentiment (Bullish/Bearish/Neutral)
            2. Key themes or trends identified (e.g., policy changes, sector performance)
            3. Potential impact on Indian stock sectors (e.g., IT, Banking, FMCG, Auto, Pharma)
            4. One actionable insight for portfolio positioning in India
            
            Keep the analysis concise and focused on investment implications for Indian investors.
            """
            try:
                response = self.gemini_model.generate_content(
                    contents=prompt,
                    generation_config={
                        'max_output_tokens': 5000,
                        'temperature': 0.3
                    },
                    request_options={'timeout': 120}
                )
                analysis = response.text
            except Exception as e:
                self.logger.error(f"Gemini news sentiment analysis failed: {str(e)}")
                return news_items  # Return raw news without sentiment

            # Append Gemini analysis to news items
            for item in news_items:
                item['overall_sentiment_score'] = 0  # Placeholder, as Gemini provides textual sentiment
                item['overall_sentiment_label'] = analysis.split('\n')[0].replace('1. Overall market sentiment: ', '') if analysis else 'Neutral'
                item['analysis'] = analysis

            self.logger.info("Successfully fetched and analyzed market news with Gemini")
            return news_items

        except Exception as e:
            st.warning(f"Could not fetch or analyze market news: {str(e)}")
            return []

    def get_sector_performance(self):
        """Get sector performance data for Indian market using yfinance"""
        try:
            # Define major Indian sectors (consistent with RiskCalculator and AIAnalyzer)
            sector_map = {
                'Nifty IT': 'NIFTYIT.NS',
                'Nifty Bank': 'BANKNIFTY.NS',
                'Nifty FMCG': 'NIFTYFMCG.NS',
                'Nifty Auto': 'NIFTYAUTO.NS',
                'Nifty Pharma': 'NIFTYPHARMA.NS'
            }
            performance = {}
            for sector, ticker in sector_map.items():
                try:
                    df = self.get_stock_data(ticker)
                    if not df.empty:
                        returns = df['close'].pct_change(periods=30).iloc[-1] * 100  # 30-day return
                        performance[sector] = round(returns, 2)
                except Exception as e:
                    self.logger.warning(f"Could not fetch data for {sector} ({ticker}): {str(e)}")
            self.logger.info("Fetched sector performance for Indian market")
            return performance

        except Exception as e:
            st.warning(f"Could not fetch sector performance: {str(e)}")
            return {}

    def calculate_correlation_matrix(self, symbols, period_days=252):
        """Calculate correlation matrix for given Indian stock symbols"""
        try:
            price_data = {}
            for symbol in symbols:
                data = self.get_stock_data(symbol)
                if data is not None and not data.empty:
                    recent_data = data.tail(period_days)
                    price_data[symbol] = recent_data['close']

            if len(price_data) < 2:
                self.logger.warning("Insufficient data for correlation matrix")
                return pd.DataFrame()

            # Create DataFrame and calculate correlation
            df = pd.DataFrame(price_data)
            df = df.dropna()
            if df.empty:
                self.logger.warning("No overlapping data for correlation matrix")
                return pd.DataFrame()

            correlation_matrix = df.corr()
            self.logger.info("Successfully calculated correlation matrix")
            return correlation_matrix

        except Exception as e:
            st.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()

    def get_volatility_data(self, symbols, period_days=30):
        """Calculate volatility for given Indian stock symbols"""
        volatility_data = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol)
                if data is not None and not data.empty:
                    recent_data = data.tail(period_days + 1)
                    returns = recent_data['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    volatility_data[symbol] = volatility
            except Exception as e:
                self.logger.warning(f"Could not calculate volatility for {symbol}: {str(e)}")
        self.logger.info("Successfully calculated volatility data")
        return volatility_data