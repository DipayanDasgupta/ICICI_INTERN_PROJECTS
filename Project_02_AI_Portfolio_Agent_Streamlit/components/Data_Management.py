import streamlit as st
import sys
import os
import logging
import pandas as pd
from datetime import datetime

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.database_manager import DatabaseManager
from utils.data_fetcher import DataFetcher
from utils.fallback_data_provider import FallbackDataProvider

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

def main():
    """Data Management interface for querying collected Indian market data"""
    logger.info("Loading India Data Management component")
    
    st.title("🇮🇳 India Data Management & Analytics")
    st.markdown("### Query and export Indian market data and portfolio insights")
    
    # Initialize database connection
    try:
        db_manager = DatabaseManager()
        if not db_manager.is_available():
            st.warning("**Database Not Available**")
            st.info("Database initialization failed. The app can still work in real-time mode.")
            st.markdown("**Available features:**")
            st.markdown("- View real-time Indian market data in Indian Market Analysis")
            st.markdown("- Use AI Portfolio Agent for portfolio analysis")
            st.markdown("- All core features work without database")
            return
        else:
            db_info = db_manager.get_database_info()
            if db_info["type"] == "sqlite":
                st.success("**SQLite Database Active** - Lightweight data caching enabled")
                st.info("💡 Upgrade tip: Set DATABASE_URL for PostgreSQL in production")
            elif db_info["type"] == "postgresql":
                st.success("**PostgreSQL Database Active** - Production-grade data storage")
            else:
                st.success("**Database Connected** - You can query collected Indian market data")
            
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        st.error(f"Database connection error: {str(e)}")
        return
    
    # Initialize DataFetcher and FallbackDataProvider
    try:
        data_fetcher = DataFetcher()
        fallback = FallbackDataProvider()
        logger.info("DataFetcher and FallbackDataProvider initialized")
    except Exception as e:
        logger.error(f"Error initializing DataFetcher/FallbackDataProvider: {str(e)}")
        st.error(f"Error initializing data components: {str(e)}")
        return
    
    # Show database statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Database Overview")
        try:
            db_info = db_manager.get_database_info()
            
            if db_info["type"] == "sqlite":
                st.success(f"✅ SQLite Database Connected")
                st.info(f"📁 File: {db_info['file_path']}")
            elif db_info["type"] == "postgresql":
                st.success(f"✅ PostgreSQL Database Connected")
                st.info(f"🔗 Production Database")
            
            st.metric("Market Data Records", db_info.get('market_records', 0))
            st.metric("Portfolio Records", db_info.get('portfolio_records', 0))
            
            assets = db_manager.get_asset_universe()
            st.metric("Assets Tracked", len(assets))
            
            if assets:
                st.markdown("**Recent Indian Assets:**")
                for asset in assets[:8]:  # Show first 8
                    st.write(f"📈 {asset['symbol']} ({asset['asset_type']}) - {asset['data_points']} records")
                    
        except Exception as e:
            logger.error(f"Error loading database info: {str(e)}")
            st.error(f"Error loading database info: {str(e)}")
    
    with col2:
        st.subheader("🔍 Quick Data Query")
        
        if st.button("📈 View Recent Indian Market Data", type="primary"):
            try:
                cursor = db_manager.conn.cursor()
                
                query = """
                    SELECT symbol, asset_type, timestamp, close_price, volume
                    FROM market_data 
                    WHERE asset_type = 'stock'
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """
                params = () if db_manager.db_type == "sqlite" else None
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    df = pd.DataFrame(results, columns=['Symbol', 'Type', 'Timestamp', 'Price (₹)', 'Volume'])
                    df['Price (₹)'] = df['Price (₹)'].apply(lambda x: f"₹{x:,.2f}")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"indian_market_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No Indian market data found in database")
                    
            except Exception as e:
                logger.error(f"Quick query error: {str(e)}")
                st.error(f"Query error: {str(e)}")
    
    # Advanced query section
    st.markdown("---")
    st.subheader("🛠️ Advanced Queries")
    
    query_type = st.selectbox(
        "Select Query Type:",
        ["Stock Data by Symbol", "Sector Index Data", "Portfolio Recommendations", "Custom SQL"]
    )
    
    if query_type == "Stock Data by Symbol":
        symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")
        
        if st.button("📊 Get Stock Data"):
            try:
                cursor = db_manager.conn.cursor()
                
                query = """
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data 
                    WHERE symbol = ? AND asset_type = 'stock'
                    ORDER BY timestamp DESC 
                    LIMIT 30
                """ if db_manager.db_type == "sqlite" else """
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data 
                    WHERE symbol = %s AND asset_type = 'stock'
                    ORDER BY timestamp DESC 
                    LIMIT 30
                """
                
                cursor.execute(query, (symbol.upper(),))
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    df = pd.DataFrame(results, columns=['Timestamp', 'Open (₹)', 'High (₹)', 'Low (₹)', 'Close (₹)', 'Volume'])
                    df[['Open (₹)', 'High (₹)', 'Low (₹)', 'Close (₹)']] = df[['Open (₹)', 'High (₹)', 'Low (₹)', 'Close (₹)']].apply(lambda x: x.apply(lambda y: f"₹{y:,.2f}"))
                    st.success(f"Found {len(df)} records for {symbol}")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {symbol} Data",
                        data=csv,
                        file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    # Try fetching real-time data as fallback
                    data = data_fetcher.get_stock_data(symbol)
                    if data.empty:
                        data = fallback.generate_stock_data(symbol, days=30)
                    if not data.empty:
                        df = pd.DataFrame({
                            'Timestamp': data.index,
                            'Open (₹)': data['open'].apply(lambda x: f"₹{x:,.2f}"),
                            'High (₹)': data['high'].apply(lambda x: f"₹{x:,.2f}"),
                            'Low (₹)': data['low'].apply(lambda x: f"₹{x:,.2f}"),
                            'Close (₹)': data['close'].apply(lambda x: f"₹{x:,.2f}"),
                            'Volume': data['volume']
                        })
                        st.success(f"Found {len(df)} real-time records for {symbol}")
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {symbol} Data",
                            data=csv,
                            file_name=f"{symbol}_realtime_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Stock data query error: {str(e)}")
                st.error(f"Query error: {str(e)}")
    
    elif query_type == "Sector Index Data":
        sector = st.selectbox(
            "Select Sector Index:",
            options=['Nifty IT', 'Nifty Bank', 'Nifty FMCG', 'Nifty Auto', 'Nifty Pharma']
        )
        sector_tickers = {
            'Nifty IT': '^CNXIT',
            'Nifty Bank': '^NSEBANK',
            'Nifty FMCG': '^CNXFMCG',
            'Nifty Auto': '^CNXAUTO',
            'Nifty Pharma': '^CNXPHARMA'
        }
        
        if st.button("📊 Get Sector Data"):
            try:
                symbol = sector_tickers[sector]
                cursor = db_manager.conn.cursor()
                
                query = """
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data 
                    WHERE symbol = ? AND asset_type = 'index'
                    ORDER BY timestamp DESC 
                    LIMIT 30
                """ if db_manager.db_type == "sqlite" else """
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data 
                    WHERE symbol = %s AND asset_type = 'index'
                    ORDER BY timestamp DESC 
                    LIMIT 30
                """
                
                cursor.execute(query, (symbol,))
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    df = pd.DataFrame(results, columns=['Timestamp', 'Open (₹)', 'High (₹)', 'Low (₹)', 'Close (₹)', 'Volume'])
                    df[['Open (₹)', 'High (₹)', 'Low (₹)', 'Close (₹)']] = df[['Open (₹)', 'High (₹)', 'Low (₹)', 'Close (₹)']].apply(lambda x: x.apply(lambda y: f"₹{y:,.2f}"))
                    st.success(f"Found {len(df)} records for {sector} ({symbol})")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {sector} Data",
                        data=csv,
                        file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    # Try fetching real-time data
                    data = data_fetcher.get_stock_data(symbol)
                    if data.empty:
                        data = fallback.generate_stock_data(symbol, days=30)
                    if not data.empty:
                        df = pd.DataFrame({
                            'Timestamp': data.index,
                            'Open (₹)': data['open'].apply(lambda x: f"₹{x:,.2f}"),
                            'High (₹)': data['high'].apply(lambda x: f"₹{x:,.2f}"),
                            'Low (₹)': data['low'].apply(lambda x: f"₹{x:,.2f}"),
                            'Close (₹)': data['close'].apply(lambda x: f"₹{x:,.2f}"),
                            'Volume': data['volume']
                        })
                        st.success(f"Found {len(df)} real-time records for {sector} ({symbol})")
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {sector} Data",
                            data=csv,
                            file_name=f"{symbol}_realtime_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No data found for {sector} ({symbol})")
                    
            except Exception as e:
                logger.error(f"Sector data query error: {str(e)}")
                st.error(f"Query error: {str(e)}")
    
    elif query_type == "Portfolio Recommendations":
        if st.button("📊 View Recent Portfolio Recommendations"):
            try:
                cursor = db_manager.conn.cursor()
                
                query = """
                    SELECT user_session, investment_amount, risk_profile, recommended_allocation, ai_analysis, created_at
                    FROM portfolio_recommendations 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """
                params = () if db_manager.db_type == "sqlite" else None
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    df = pd.DataFrame(results, columns=['Session', 'Amount (₹)', 'Risk Profile', 'Allocation', 'Analysis', 'Created At'])
                    df['Amount (₹)'] = df['Amount (₹)'].apply(lambda x: f"₹{x:,.2f}")
                    df['Allocation'] = df['Allocation'].apply(lambda x: json.dumps(json.loads(x), indent=2) if x else '{}')
                    st.success(f"Found {len(df)} portfolio recommendations")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Portfolio Recommendations",
                        data=csv,
                        file_name=f"portfolio_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No portfolio recommendations found in database")
                    
            except Exception as e:
                logger.error(f"Portfolio recommendations query error: {str(e)}")
                st.error(f"Query error: {str(e)}")
    
    elif query_type == "Custom SQL":
        st.markdown("**Execute Custom SQL Query**")
        st.warning("⚠️ Only SELECT queries allowed for security")
        
        sql_query = st.text_area(
            "Enter SQL Query:",
            placeholder="SELECT * FROM market_data WHERE symbol = 'RELIANCE.NS' LIMIT 10;",
            height=100
        )
        
        if st.button("🚀 Execute Query"):
            if sql_query.strip():
                try:
                    if not sql_query.upper().strip().startswith('SELECT'):
                        st.error("Only SELECT queries are allowed")
                    else:
                        cursor = db_manager.conn.cursor()
                        cursor.execute(sql_query)
                        results = cursor.fetchall()
                        
                        if results:
                            colnames = [desc[0] for desc in cursor.description]
                            cursor.close()
                            
                            df = pd.DataFrame(results, columns=colnames)
                            if 'close_price' in df.columns:
                                df['close_price'] = df['close_price'].apply(lambda x: f"₹{x:,.2f}")
                            if 'open_price' in df.columns:
                                df['open_price'] = df['open_price'].apply(lambda x: f"₹{x:,.2f}")
                            if 'high_price' in df.columns:
                                df['high_price'] = df['high_price'].apply(lambda x: f"₹{x:,.2f}")
                            if 'low_price' in df.columns:
                                df['low_price'] = df['low_price'].apply(lambda x: f"₹{x:,.2f}")
                            st.success(f"Query returned {len(df)} records")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("Query executed successfully but returned no results")
                            
                except Exception as e:
                    logger.error(f"Custom SQL query error: {str(e)}")
                    st.error(f"Query error: {str(e)}")
    
    # Show recent activity
    st.markdown("---")
    st.subheader("📈 Recent Database Activity")
    
    try:
        cursor = db_manager.conn.cursor()
        
        query = """
            SELECT DISTINCT symbol, asset_type, 
                   COUNT(*) as records,
                   MAX(created_at) as last_updated
            FROM market_data 
            WHERE asset_type IN ('stock', 'index')
            GROUP BY symbol, asset_type
            ORDER BY last_updated DESC
            LIMIT 10
        """
        params = () if db_manager.db_type == "sqlite" else None
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        
        if results:
            activity_df = pd.DataFrame(results, columns=['Symbol', 'Type', 'Records', 'Last Updated'])
            st.dataframe(activity_df, use_container_width=True)
        else:
            st.info("No recent activity found")
            
    except Exception as e:
        logger.error(f"Error loading activity: {str(e)}")
        st.error(f"Error loading activity: {str(e)}")

if __name__ == "__main__":
    main()