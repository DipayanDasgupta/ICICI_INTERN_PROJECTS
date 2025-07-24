import streamlit as st
import sys
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_fetcher import DataFetcher
from utils.ai_analyzer import AIAnalyzer
from utils.database_manager import DatabaseManager
from utils.fallback_data_provider import FallbackDataProvider
from utils.risk_calculator import RiskCalculator

# Set up logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('portfolio_app_india.log')  # Aligned with main app
    ]
)
logger = logging.getLogger(__name__)

def init_components():
    """Initialize all required components with logging"""
    logger.info("Initializing India Market Analysis components...")
    
    try:
        if 'data_fetcher' not in st.session_state:
            logger.info("Creating DataFetcher for Market Analysis")
            st.session_state.data_fetcher = DataFetcher()
        
        if 'ai_analyzer' not in st.session_state:
            logger.info("Creating AIAnalyzer for Market Analysis")
            st.session_state.ai_analyzer = AIAnalyzer()
        
        if 'db_manager' not in st.session_state:
            logger.info("Creating DatabaseManager for Market Analysis")
            st.session_state.db_manager = DatabaseManager()
        
        if 'fallback_data_provider' not in st.session_state:
            logger.info("Creating FallbackDataProvider for Market Analysis")
            st.session_state.fallback_data_provider = FallbackDataProvider()
        
        if 'risk_calculator' not in st.session_state:
            logger.info("Creating RiskCalculator for Market Analysis")
            symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            st.session_state.risk_calculator = RiskCalculator(symbols=symbols, market_index='^NSEI')
        
        logger.info("Market Analysis components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Market Analysis components: {str(e)}")
        st.error(f"Initialization error: {str(e)}")

def main():
    logger.info("Starting India Market Analysis main function")
    
    try:
        # Initialize components
        init_components()
        
        data_fetcher = st.session_state.data_fetcher
        ai_analyzer = st.session_state.ai_analyzer
        db = st.session_state.db_manager
        fallback = st.session_state.fallback_data_provider
        risk_calc = st.session_state.risk_calculator
        
        logger.info("All components loaded successfully for Market Analysis")
    except Exception as e:
        logger.error(f"Error initializing Market Analysis components: {str(e)}")
        st.error(f"Initialization error: {str(e)}")
        return
    
    st.title("📈 India Market Analysis")
    st.markdown("### Real-time insights for Indian equities and sector indices")
    
    # Market overview section
    st.subheader("🇮🇳 Indian Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📈 Major Indices**")
        try:
            logger.info("Fetching Nifty 50 index data")
            nifty_data = data_fetcher.get_stock_data('^NSEI')
            if nifty_data is not None and not nifty_data.empty:
                nifty_current = float(nifty_data.iloc[-1]['close'])
                nifty_prev = float(nifty_data.iloc[-2]['close']) if len(nifty_data) > 1 else nifty_current
                nifty_change = ((nifty_current - nifty_prev) / nifty_prev) * 100 if nifty_prev != 0 else 0
                st.metric("Nifty 50", f"₹{nifty_current:,.2f}", f"{nifty_change:+.2f}%")
                logger.info(f"Successfully displayed Nifty 50: ₹{nifty_current:,.2f}")
            else:
                nifty_data = fallback.generate_stock_data('^NSEI', days=2)
                if not nifty_data.empty:
                    nifty_current = float(nifty_data.iloc[-1]['close'])
                    st.metric("Nifty 50", f"₹{nifty_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for Nifty 50")
                else:
                    st.metric("Nifty 50", "Data unavailable", None)
                    logger.warning("Nifty 50 data unavailable")
            
            logger.info("Fetching Sensex index data")
            sensex_data = data_fetcher.get_stock_data('^BSESN')
            if sensex_data is not None and not sensex_data.empty:
                sensex_current = float(sensex_data.iloc[-1]['close'])
                sensex_prev = float(sensex_data.iloc[-2]['close']) if len(sensex_data) > 1 else sensex_current
                sensex_change = ((sensex_current - sensex_prev) / sensex_prev) * 100 if sensex_prev != 0 else 0
                st.metric("Sensex", f"₹{sensex_current:,.2f}", f"{sensex_change:+.2f}%")
                logger.info(f"Successfully displayed Sensex: ₹{sensex_current:,.2f}")
            else:
                sensex_data = fallback.generate_stock_data('^BSESN', days=2)
                if not sensex_data.empty:
                    sensex_current = float(sensex_data.iloc[-1]['close'])
                    st.metric("Sensex", f"₹{sensex_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for Sensex")
                else:
                    st.metric("Sensex", "Data unavailable", None)
                    logger.warning("Sensex data unavailable")
        except Exception as e:
            st.error(f"Index data temporarily unavailable")
            logger.error(f"Error loading index data: {str(e)}", exc_info=True)
    
    with col2:
        st.markdown("**🏦 Banking Sector**")
        try:
            banknifty_data = data_fetcher.get_stock_data('^NSEBANK')
            if banknifty_data is not None and not banknifty_data.empty:
                banknifty_current = float(banknifty_data.iloc[-1]['close'])
                banknifty_prev = float(banknifty_data.iloc[-2]['close']) if len(banknifty_data) > 1 else banknifty_current
                banknifty_change = ((banknifty_current - banknifty_prev) / banknifty_prev) * 100 if banknifty_prev != 0 else 0
                st.metric("Nifty Bank", f"₹{banknifty_current:,.2f}", f"{banknifty_change:+.2f}%")
                logger.info(f"Successfully displayed Nifty Bank: ₹{banknifty_current:,.2f}")
            else:
                banknifty_data = fallback.generate_stock_data('^NSEBANK', days=2)
                if not banknifty_data.empty:
                    banknifty_current = float(banknifty_data.iloc[-1]['close'])
                    st.metric("Nifty Bank", f"₹{banknifty_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for Nifty Bank")
                else:
                    st.metric("Nifty Bank", "Data unavailable", None)
                    logger.warning("Nifty Bank data unavailable")
            
            hdfcbank_data = data_fetcher.get_stock_data('HDFCBANK.NS')
            if hdfcbank_data is not None and not hdfcbank_data.empty:
                hdfcbank_current = float(hdfcbank_data.iloc[-1]['close'])
                hdfcbank_prev = float(hdfcbank_data.iloc[-2]['close']) if len(hdfcbank_data) > 1 else hdfcbank_current
                hdfcbank_change = ((hdfcbank_current - hdfcbank_prev) / hdfcbank_prev) * 100 if hdfcbank_prev != 0 else 0
                st.metric("HDFC Bank", f"₹{hdfcbank_current:,.2f}", f"{hdfcbank_change:+.2f}%")
                logger.info(f"Successfully displayed HDFC Bank: ₹{hdfcbank_current:,.2f}")
            else:
                hdfcbank_data = fallback.generate_stock_data('HDFCBANK.NS', days=2)
                if not hdfcbank_data.empty:
                    hdfcbank_current = float(hdfcbank_data.iloc[-1]['close'])
                    st.metric("HDFC Bank", f"₹{hdfcbank_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for HDFC Bank")
                else:
                    st.metric("HDFC Bank", "Data unavailable", None)
                    logger.warning("HDFC Bank data unavailable")
        except Exception as e:
            st.error(f"Banking sector data temporarily unavailable")
            logger.error(f"Error loading banking sector data: {str(e)}", exc_info=True)
    
    with col3:
        st.markdown("**💻 IT Sector**")
        try:
            niftyit_data = data_fetcher.get_stock_data('^CNXIT')
            if niftyit_data is not None and not niftyit_data.empty:
                niftyit_current = float(niftyit_data.iloc[-1]['close'])
                niftyit_prev = float(niftyit_data.iloc[-2]['close']) if len(niftyit_data) > 1 else niftyit_current
                niftyit_change = ((niftyit_current - niftyit_prev) / niftyit_prev) * 100 if niftyit_prev != 0 else 0
                st.metric("Nifty IT", f"₹{niftyit_current:,.2f}", f"{niftyit_change:+.2f}%")
                logger.info(f"Successfully displayed Nifty IT: ₹{niftyit_current:,.2f}")
            else:
                niftyit_data = fallback.generate_stock_data('^CNXIT', days=2)
                if not niftyit_data.empty:
                    niftyit_current = float(niftyit_data.iloc[-1]['close'])
                    st.metric("Nifty IT", f"₹{niftyit_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for Nifty IT")
                else:
                    st.metric("Nifty IT", "Data unavailable", None)
                    logger.warning("Nifty IT data unavailable")
            
            tcs_data = data_fetcher.get_stock_data('TCS.NS')
            if tcs_data is not None and not tcs_data.empty:
                tcs_current = float(tcs_data.iloc[-1]['close'])
                tcs_prev = float(tcs_data.iloc[-2]['close']) if len(tcs_data) > 1 else tcs_current
                tcs_change = ((tcs_current - tcs_prev) / tcs_prev) * 100 if tcs_prev != 0 else 0
                st.metric("TCS", f"₹{tcs_current:,.2f}", f"{tcs_change:+.2f}%")
                logger.info(f"Successfully displayed TCS: ₹{tcs_current:,.2f}")
            else:
                tcs_data = fallback.generate_stock_data('TCS.NS', days=2)
                if not tcs_data.empty:
                    tcs_current = float(tcs_data.iloc[-1]['close'])
                    st.metric("TCS", f"₹{tcs_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for TCS")
                else:
                    st.metric("TCS", "Data unavailable", None)
                    logger.warning("TCS data unavailable")
        except Exception as e:
            st.error(f"IT sector data temporarily unavailable")
            logger.error(f"Error loading IT sector data: {str(e)}", exc_info=True)
    
    with col4:
        st.markdown("**🛒 FMCG Sector**")
        try:
            niftyfmcg_data = data_fetcher.get_stock_data('^CNXFMCG')
            if niftyfmcg_data is not None and not niftyfmcg_data.empty:
                niftyfmcg_current = float(niftyfmcg_data.iloc[-1]['close'])
                niftyfmcg_prev = float(niftyfmcg_data.iloc[-2]['close']) if len(niftyfmcg_data) > 1 else niftyfmcg_current
                niftyfmcg_change = ((niftyfmcg_current - niftyfmcg_prev) / niftyfmcg_prev) * 100 if niftyfmcg_prev != 0 else 0
                st.metric("Nifty FMCG", f"₹{niftyfmcg_current:,.2f}", f"{niftyfmcg_change:+.2f}%")
                logger.info(f"Successfully displayed Nifty FMCG: ₹{niftyfmcg_current:,.2f}")
            else:
                niftyfmcg_data = fallback.generate_stock_data('^CNXFMCG', days=2)
                if not niftyfmcg_data.empty:
                    niftyfmcg_current = float(niftyfmcg_data.iloc[-1]['close'])
                    st.metric("Nifty FMCG", f"₹{niftyfmcg_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for Nifty FMCG")
                else:
                    st.metric("Nifty FMCG", "Data unavailable", None)
                    logger.warning("Nifty FMCG data unavailable")
            
            hindunilvr_data = data_fetcher.get_stock_data('HINDUNILVR.NS')
            if hindunilvr_data is not None and not hindunilvr_data.empty:
                hindunilvr_current = float(hindunilvr_data.iloc[-1]['close'])
                hindunilvr_prev = float(hindunilvr_data.iloc[-2]['close']) if len(hindunilvr_data) > 1 else hindunilvr_current
                hindunilvr_change = ((hindunilvr_current - hindunilvr_prev) / hindunilvr_prev) * 100 if hindunilvr_prev != 0 else 0
                st.metric("Hindustan Unilever", f"₹{hindunilvr_current:,.2f}", f"{hindunilvr_change:+.2f}%")
                logger.info(f"Successfully displayed Hindustan Unilever: ₹{hindunilvr_current:,.2f}")
            else:
                hindunilvr_data = fallback.generate_stock_data('HINDUNILVR.NS', days=2)
                if not hindunilvr_data.empty:
                    hindunilvr_current = float(hindunilvr_data.iloc[-1]['close'])
                    st.metric("Hindustan Unilever", f"₹{hindunilvr_current:,.2f}", "Synthetic data")
                    logger.warning("Using synthetic data for Hindustan Unilever")
                else:
                    st.metric("Hindustan Unilever", "Data unavailable", None)
                    logger.warning("Hindustan Unilever data unavailable")
        except Exception as e:
            st.error(f"FMCG sector data temporarily unavailable")
            logger.error(f"Error loading FMCG sector data: {str(e)}", exc_info=True)
    
    st.markdown("---")
    
    # Economic indicators section
    st.subheader("📈 Indian Economic Indicators")
    
    col_econ1, col_econ2 = st.columns(2)
    
    with col_econ1:
        st.markdown("**Key Economic Data**")
        try:
            # Fetch or use cached economic indicators from DatabaseManager
            if db.is_available():
                cursor = db.conn.cursor()
                query = """
                    SELECT indicator_name, value, change
                    FROM economic_indicators
                    WHERE country = 'India'
                    ORDER BY updated_at DESC
                    LIMIT 6
                """
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    for indicator, value, change in results:
                        st.metric(indicator, value, f"{float(change):+.2f}%")
                else:
                    # Fallback to static or real-time data
                    st.metric("RBI Repo Rate", "6.50%", "0.00%")
                    st.metric("CPI Inflation", "5.1%", "-0.2%")
                    st.metric("Unemployment Rate", "7.8%", "+0.1%")
                    st.metric("GDP Growth (YoY)", "7.6%", "+0.4%")
                    st.metric("INR/USD", "83.50", "+0.10")
                    st.metric("India VIX", "14.5", "-1.2")
                    logger.info("Displayed fallback economic indicators")
            else:
                st.metric("RBI Repo Rate", "6.50%", "0.00%")
                st.metric("CPI Inflation", "5.1%", "-0.2%")
                st.metric("Unemployment Rate", "7.8%", "+0.1%")
                st.metric("GDP Growth (YoY)", "7.6%", "+0.4%")
                st.metric("INR/USD", "83.50", "+0.10")
                st.metric("India VIX", "14.5", "-1.2")
                logger.info("Displayed static economic indicators due to database unavailability")
        except Exception as e:
            logger.error(f"Error displaying economic indicators: {str(e)}")
            st.warning("Economic data temporarily unavailable")
    
    with col_econ2:
        st.markdown("**Indian Market News Summary**")
        try:
            # Fetch news or analysis from AIAnalyzer
            news_summary = ai_analyzer.get_market_analysis("Summarize recent Indian market news affecting Nifty 50 and key sectors")
            if news_summary and news_summary != "AI analysis not available":
                st.write(news_summary)
                # Store in DatabaseManager
                if db.is_available():
                    cursor = db.conn.cursor()
                    query = """
                        INSERT INTO market_news (timestamp, summary, source)
                        VALUES (?, ?, ?)
                    """ if db.db_type == "sqlite" else """
                        INSERT INTO market_news (timestamp, summary, source)
                        VALUES (%s, %s, %s)
                    """
                    cursor.execute(query, (datetime.now().isoformat(), news_summary, 'AIAnalyzer'))
                    db.conn.commit()
                    cursor.close()
            else:
                # Fallback to static news
                st.write("**🏦 RBI Maintains Repo Rate at 6.50%**")
                st.write("The Reserve Bank of India continues its balanced approach to support growth while monitoring inflation...")
                st.markdown("---")
                st.write("**💻 IT Sector Gains Momentum**")
                st.write("Indian IT stocks rally due to strong global demand for digital services...")
                st.markdown("---")
                st.write("**🛒 FMCG Sector Stable**")
                st.write("FMCG companies show resilience amid steady consumer demand...")
                logger.info("Displayed fallback market news")
        except Exception as e:
            logger.error(f"Error displaying market news: {str(e)}")
            st.warning("News data temporarily unavailable")
    
    st.markdown("---")
    
    # AI Market Analysis
    st.subheader("🧠 AI-Powered Indian Market Analysis")
    
    if st.button("🚀 Generate Comprehensive Indian Market Analysis", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing Indian markets and sectors..."):
            try:
                market_context = {
                    'timestamp': datetime.now().isoformat(),
                    'asset_classes': ['equity'],
                    'indices': ['^NSEI', '^BSESN', 'NIFTYIT.NS', 'BANKNIFTY.NS', 'NIFTYFMCG.NS'],
                    'stocks': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS']
                }
                analysis = ai_analyzer.get_market_analysis(f"Analyze the Indian stock market, focusing on Nifty 50, Sensex, and key sectors: {market_context['indices']}")
                
                if analysis and analysis != "AI analysis not available":
                    st.success("✅ Analysis Complete")
                    st.markdown("### AI-Powered Indian Market Outlook")
                    st.write(analysis)
                    
                    # Store in DatabaseManager
                    if db.is_available():
                        cursor = db.conn.cursor()
                        query = """
                            INSERT INTO market_analysis (timestamp, analysis, source)
                            VALUES (?, ?, ?)
                        """ if db.db_type == "sqlite" else """
                            INSERT INTO market_analysis (timestamp, analysis, source)
                            VALUES (%s, %s, %s)
                        """
                        cursor.execute(query, (datetime.now().isoformat(), analysis, 'AIAnalyzer'))
                        db.conn.commit()
                        cursor.close()
                    
                    st.session_state.latest_market_analysis = analysis
                else:
                    st.error("Market analysis temporarily unavailable")
                    logger.warning("AI market analysis returned empty or unavailable")
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
                logger.error(f"Error generating AI market analysis: {str(e)}")
    
    # Display cached analysis if available
    if 'latest_market_analysis' in st.session_state:
        with st.expander("📋 Latest Market Analysis"):
            st.write(st.session_state.latest_market_analysis)
    
    st.markdown("---")
    
    # Sector performance
    st.subheader("🏢 Indian Sector Performance")
    
    try:
        logger.info("Fetching Indian sector performance data")
        sectors = {
            'Nifty IT': '^CNXIT',
            'Nifty Bank': '^NSEBANK',
            'Nifty FMCG': '^CNXFMCG',
            'Nifty Auto': '^CNXAUTO',
            'Nifty Pharma': '^CNXPHARMA'
        }
        
        sector_performance = {}
        for sector_name, ticker in sectors.items():
            try:
                data = data_fetcher.get_stock_data(ticker)
                if data.empty:
                    data = fallback.generate_stock_data(ticker, days=2)
                if not data.empty and len(data) > 1:
                    current_price = float(data['close'].iloc[-1])
                    prev_price = float(data['close'].iloc[-2])
                    performance = ((current_price - prev_price) / prev_price) * 100
                    sector_performance[sector_name] = round(performance, 2)
                    logger.info(f"Updated {sector_name} with real data: {performance:.2f}%")
                else:
                    sector_performance[sector_name] = 0.0
                    logger.warning(f"No data for {sector_name} ({ticker})")
            except Exception as e:
                sector_performance[sector_name] = 0.0
                logger.warning(f"Error fetching data for {sector_name} ({ticker}): {str(e)}")
        
        # Create sector performance chart
        sector_names = list(sector_performance.keys())
        performances = list(sector_performance.values())
        
        fig = px.bar(
            x=performances,
            y=sector_names,
            orientation='h',
            title="Indian Sector Performance Today",
            labels={'x': 'Performance (%)', 'y': 'Sectors'},
            color=performances,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sector details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Top Performers**")
            top_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)[:3]
            for sector, perf in top_sectors:
                st.write(f"📈 {sector}: +{perf:.1f}%")
        
        with col2:
            st.markdown("**Bottom Performers**")
            bottom_sectors = sorted(sector_performance.items(), key=lambda x: x[1])[:3]
            for sector, perf in bottom_sectors:
                st.write(f"📉 {sector}: {perf:.1f}%")
        
        with col3:
            st.markdown("**Market Notes**")
            st.write("• IT sector driven by global demand")
            st.write("• Banking stable with RBI policy")
            st.write("• FMCG resilient amid inflation")
        
    except Exception as e:
        logger.error(f"Error loading sector performance: {str(e)}")
        st.error("Unable to load sector performance data")
    
    # Database status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Database Status")
    
    try:
        if db.is_available():
            asset_count = len(db.get_asset_universe())
            st.sidebar.metric("Assets Tracked", asset_count)
            st.sidebar.success("Database Connected")
        else:
            st.sidebar.error("Database Offline")
    except Exception as e:
        st.sidebar.error(f"Database Error: {str(e)}")

if __name__ == "__main__":
    main()