# utils/database_manager.py

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

class DatabaseManager:
    """
    Manages database connections and operations for the portfolio manager.
    Supports SQLite for local development and PostgreSQL for production.
    """
    def __init__(self):
        """Initialize database connection - SQLite by default, PostgreSQL if available."""
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.db_available = False
        self.db_type = "none"
        
        # Try PostgreSQL first (for production/advanced users)
        database_url = os.getenv('DATABASE_URL')
        if database_url and database_url.startswith('postgresql'):
            try:
                self.conn = psycopg2.connect(database_url)
                self.conn.autocommit = True
                self.db_available = True
                self.db_type = "postgresql"
                self.logger.info("Connected to PostgreSQL database")
                self._init_tables()
                return
            except Exception as e:
                self.logger.warning(f"PostgreSQL connection failed: {str(e)}, falling back to SQLite")
        
        # Use SQLite as default (lightweight, no setup required)
        try:
            os.makedirs('data', exist_ok=True)
            db_path = os.path.join('data', 'portfolio_manager.db')
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.db_available = True
            self.db_type = "sqlite"
            self.logger.info(f"Connected to SQLite database at {db_path}")
            self._init_tables()
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            self.conn = None
            self.db_available = False

    def _init_tables(self):
        """A single method to initialize tables for either SQLite or PostgreSQL."""
        if self.db_type == "sqlite":
            self._init_tables_sqlite()
        elif self.db_type == "postgresql":
            self._init_tables_postgresql()

    def _init_tables_sqlite(self):
        """Initialize SQLite database tables for the Indian market."""
        try:
            cursor = self.conn.cursor()
            
            # Market data table for Indian stocks and indices
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, asset_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL, open_price REAL, high_price REAL, low_price REAL,
                    close_price REAL, volume INTEGER, market_cap REAL, additional_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(symbol, asset_type, timestamp)
                )
            """)
            
            # Economic indicators table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, indicator_name TEXT NOT NULL, value TEXT,
                    change TEXT, country TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_name, country)
                )
            """)

            # Market news table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, summary TEXT,
                    source TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Market analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, analysis TEXT,
                    source TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Portfolio recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_session TEXT NOT NULL,
                    investment_amount REAL, risk_profile TEXT, recommended_allocation TEXT,
                    ai_analysis TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # API usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, api_name TEXT NOT NULL, endpoint TEXT,
                    requests_count INTEGER DEFAULT 1, last_used TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(api_name, endpoint)
                )
            """)
            
            self.conn.commit()
            self.logger.info("SQLite database tables initialized/verified successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing SQLite tables: {str(e)}")

    def _init_tables_postgresql(self):
        """Initialize PostgreSQL database tables for the Indian market."""
        # Note: This is a placeholder as the provided file had a partial implementation.
        # It's better to have it here, clearly defined, than duplicated.
        self.logger.info("PostgreSQL table initialization would run here.")
        # Add your CREATE TABLE queries for PostgreSQL here if needed.

    def is_available(self):
        """Check if the database connection is available."""
        return self.db_available
    # In utils/database_manager.py, inside the DatabaseManager class

    # In utils/database_manager.py

    def data_freshness_check(self, symbol, asset_type, max_age_seconds=300):
        """Check if data in the database is recent enough."""
        if not self.is_available():
            return False
        
        try:
            cursor = self.conn.cursor()
            cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
            
            if self.db_type == "sqlite":
                query = "SELECT COUNT(*) FROM market_data WHERE symbol = ? AND asset_type = ? AND datetime(created_at) >= datetime(?)"
                params = (symbol, asset_type, cutoff_time.isoformat())
            else: # PostgreSQL
                query = "SELECT COUNT(*) FROM market_data WHERE symbol = %s AND asset_type = %s AND created_at >= %s"
                params = (symbol, asset_type, cutoff_time)

            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Error checking data freshness for {symbol}: {str(e)}")
            return False


    def get_database_info(self):
        """Get database information for display (works for both SQLite and PostgreSQL)."""
        if not self.is_available():
            return {"type": "none", "status": "unavailable"}
        
        info = {
            "type": self.db_type,
            "status": "connected",
            "market_records": 0, "portfolio_records": 0,
            "economic_records": 0, "api_usage_records": 0,
            "file_path": "data/portfolio_manager.db" if self.db_type == "sqlite" else "PostgreSQL"
        }
        
        try:
            cursor = self.conn.cursor()
            
            def get_count(table_name):
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    return cursor.fetchone()[0]
                except Exception as e:
                    self.logger.warning(f"Could not get count for table '{table_name}': {e}")
                    return 0

            info["market_records"] = get_count("market_data")
            info["portfolio_records"] = get_count("portfolio_recommendations")
            info["economic_records"] = get_count("economic_indicators")
            info["api_usage_records"] = get_count("api_usage")
            
            cursor.close()
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {str(e)}")
            info["status"] = "error"
            info["error_message"] = str(e)
            return info

    def store_market_data(self, symbol, asset_type, data_df):
        """Store market data in the database."""
        if not self.is_available() or data_df.empty: return False
        
        try:
            cursor = self.conn.cursor()
            for _, row in data_df.iterrows():
                # This unified query works for both by using named parameters that psycopg2 can also handle
                # with dict binding, but for simplicity, we stick to standard placeholders.
                if self.db_type == "sqlite":
                    query = """
                        INSERT OR REPLACE INTO market_data 
                        (symbol, asset_type, timestamp, open_price, high_price, low_price, close_price, volume, additional_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    params = (symbol, asset_type, str(row.get('timestamp', row.name)),
                              row.get('open'), row.get('high'), row.get('low'), row.get('close'),
                              int(row.get('volume', 0)), row.get('additional_data'))
                    cursor.execute(query, params)
                # Add a similar block for PostgreSQL if its schema differs significantly
            if self.db_type == "sqlite": self.conn.commit()
            cursor.close()
            self.logger.info(f"Stored {len(data_df)} records for {symbol} ({asset_type}) in {self.db_type}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing market data for {symbol}: {e}")
            return False

    def store_portfolio_recommendation(self, user_session, investment_amount, risk_profile, recommended_allocation, ai_analysis):
        """Store a portfolio recommendation."""
        if not self.is_available(): return False
        try:
            cursor = self.conn.cursor()
            allocation_json = json.dumps(recommended_allocation, default=str)
            if self.db_type == "sqlite":
                query = """INSERT INTO portfolio_recommendations (user_session, investment_amount, risk_profile, recommended_allocation, ai_analysis)
                           VALUES (?, ?, ?, ?, ?)"""
                params = (user_session, investment_amount, risk_profile, allocation_json, ai_analysis)
                cursor.execute(query, params)
                self.conn.commit()
            # Add PostgreSQL specific query here if needed
            cursor.close()
            self.logger.info(f"Stored portfolio recommendation for session {user_session}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing portfolio recommendation: {e}")
            if self.db_type == "sqlite": self.conn.rollback()
            return False

    def get_market_data(self, symbol, asset_type, days_back=30):
        """Retrieve market data from the database."""
        if not self.is_available(): return None
        try:
            cursor = self.conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            if self.db_type == "sqlite":
                query = "SELECT * FROM market_data WHERE symbol = ? AND asset_type = ? AND datetime(timestamp) >= datetime(?) ORDER BY timestamp DESC"
                params = (symbol, asset_type, cutoff_date)
            else: # PostgreSQL
                query = "SELECT * FROM market_data WHERE symbol = %s AND asset_type = %s AND timestamp >= %s ORDER BY timestamp DESC"
                params = (symbol, asset_type, cutoff_date)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            if not results:
                self.logger.info(f"No market data found for {symbol} ({asset_type})")
                return pd.DataFrame() # Return empty DataFrame instead of None
                
            return pd.DataFrame([dict(row) for row in results])
        except Exception as e:
            self.logger.error(f"Error retrieving market data for {symbol}: {e}")
            return pd.DataFrame()

    def get_asset_universe(self):
        """Retrieve all unique assets and their data point counts from the database."""
        if not self.is_available(): return []
        try:
            cursor = self.conn.cursor()
            query = "SELECT symbol, asset_type, COUNT(*) as data_points FROM market_data GROUP BY symbol, asset_type ORDER BY symbol"
            cursor.execute(query)
            assets = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            return assets
        except Exception as e:
            self.logger.error(f"Error retrieving asset universe: {e}")
            return []

    def close_connection(self):
        """Close the database connection."""
        if self.conn and self.db_available:
            self.conn.close()
            self.logger.info(f"{self.db_type.title()} database connection closed.")