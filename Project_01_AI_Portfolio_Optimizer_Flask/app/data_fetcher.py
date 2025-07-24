# app/data_fetcher.py (Corrected and Enhanced for Custom Universes)

import pandas as pd
import sqlite3
import json

# Import necessary config variables
from .config import DB_FILE, STOCK_UNIVERSES, PORTFOLIOS_DB_FILE

def get_stock_universe(universe_name="NIFTY_50"):
    """
    Fetches a stock universe. If the name starts with 'custom_', it queries
    the database for a user-defined universe. Otherwise, it uses the static
    lists from the config file.
    """
    if universe_name and universe_name.startswith('custom_'):
        try:
            # The format is 'custom_ID', e.g., 'custom_1'
            universe_id = int(universe_name.split('_')[1])
            with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
                cursor = conn.cursor()
                res = cursor.execute(
                    "SELECT symbols_json FROM custom_universes WHERE id = ?",
                    (universe_id,)
                ).fetchone()
                
                if res and res[0]:
                    # The database stores the symbols as a JSON string list
                    return json.loads(res[0])
                else:
                    print(f"Warning: Custom universe with ID {universe_id} not found in database.")
                    return []
        except (ValueError, IndexError, sqlite3.Error) as e:
            print(f"Error fetching custom universe '{universe_name}': {e}")
            return []
    else:
        # Fallback to the original method for static, hardcoded universes
        return STOCK_UNIVERSES.get(universe_name, [])

def get_historical_data(symbol, start_date, end_date):
    """
    Fetches raw historical data from the database, ensuring correct data types.
    """
    start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            query = "SELECT * FROM historical_prices WHERE Symbol = ? AND Date BETWEEN ? AND ?"
            stock_df = pd.read_sql_query(query, conn, params=(symbol, start_str, end_str))

            if stock_df.empty:
                return pd.DataFrame()

            # --- DATA CLEANING AND TYPE CONVERSION ---
            # Set the Date index first
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            stock_df.set_index('Date', inplace=True)

            # Define columns that should be numeric
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in stock_df.columns:
                    # Convert to numeric, coercing errors to NaN (Not a Number)
                    stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
            
            # --- METADATA ENRICHMENT ---
            meta_query = "SELECT Sector FROM stock_metadata WHERE Symbol = ?"
            cursor = conn.cursor()
            result = cursor.execute(meta_query, (symbol,)).fetchone()
            stock_df['Sector'] = result[0] if result else 'Unknown'
            
            # Final cleanup
            stock_df.drop(columns=['Symbol'], inplace=True, errors='ignore')
            return stock_df

    except Exception as e:
        print(f"--> DATABASE ERROR fetching data for {symbol}: {e}")
        return pd.DataFrame()