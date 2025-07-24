# data_ingestion.py (Final, Most Robust Version)

import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os

from app.config import STOCK_UNIVERSES, DB_FILE, PORTFOLIOS_DB_FILE
from app.factor_analysis import ingest_fama_french_data

TEN_YEARS_AGO = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
TODAY = datetime.now().strftime('%Y-%m-%d')

def create_database():
    """Creates the databases and tables if they don't exist."""
    print(f"--- Ensuring database '{DB_FILE}' and tables exist... ---")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                Date TEXT NOT NULL,
                Symbol TEXT NOT NULL,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER,
                PRIMARY KEY (Date, Symbol)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON historical_prices (Symbol)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_metadata (
                Symbol TEXT PRIMARY KEY,
                Sector TEXT
            )
        """)
        conn.commit()
    print("--- Market data database is ready. ---")

    print(f"--- Ensuring portfolios database '{PORTFOLIOS_DB_FILE}' and tables exist... ---")
    with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                stocks_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_universes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                symbols_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print("--- Portfolios database is ready. ---")

def ingest_data():
    """
    Fetches 10 years of data, intelligently skipping fully downloaded symbols
    and robustly handling data column issues.
    """
    symbols_to_ingest = sorted(list(set(STOCK_UNIVERSES.get("NIFTY_500", []))))
    
    if '^NSEI' not in symbols_to_ingest:
        all_symbols_to_ingest = symbols_to_ingest + ['^NSEI']
    else:
        all_symbols_to_ingest = symbols_to_ingest

    print(f"\n--- Starting/Resuming data ingestion for {len(all_symbols_to_ingest)} symbols... ---")
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT Symbol FROM stock_metadata")
        existing_symbols = {row[0] for row in cursor.fetchall()}
        print(f"--- Found {len(existing_symbols)} symbols already in the database. Will skip them. ---")
        
        symbols_to_process = [s for s in all_symbols_to_ingest if s not in existing_symbols]
        print(f"--- Attempting to process {len(symbols_to_process)} new or incomplete symbols. ---")

        for symbol in tqdm(symbols_to_process, desc="Ingesting New Symbols"):
            for attempt in range(3):
                try:
                    ticker = f"{symbol}.NS" if symbol != '^NSEI' else symbol
                    data = yf.download(ticker, start=TEN_YEARS_AGO, end=TODAY, auto_adjust=False, progress=False, timeout=30)
                    
                    if data.empty:
                        tqdm.write(f"--> No data for '{symbol}'. Marking as 'No Data' and skipping.")
                        if symbol != '^NSEI':
                            cursor.execute("INSERT OR IGNORE INTO stock_metadata (Symbol, Sector) VALUES (?, ?)", (symbol, 'No Data'))
                            conn.commit()
                        break 

                    data.reset_index(inplace=True)
                    
                    # --- THIS IS THE NEW ROBUST FIX for 'duplicate column' error ---
                    # Prioritize 'Adj Close' for the 'Close' price to handle splits/dividends.
                    # This safely overwrites the 'Close' column if 'Adj Close' exists.
                    if 'Adj Close' in data.columns:
                        data['Close'] = data['Adj Close']
                    # --- END OF FIX ---
                    
                    # Now, select only the columns we need for the database.
                    # This implicitly drops the original 'Adj Close' and any other extra columns.
                    final_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
                    data['Symbol'] = symbol
                    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
                    
                    if not all(col in data.columns for col in final_columns):
                        tqdm.write(f"--> Missing critical columns for '{symbol}'. Skipping.")
                        continue

                    prices_df = data[final_columns].dropna()

                    # Use a temporary table and an INSERT OR IGNORE statement for robust insertion
                    prices_df.to_sql('temp_prices', conn, if_exists='replace', index=False)
                    cursor.execute("INSERT OR IGNORE INTO historical_prices SELECT * FROM temp_prices")
                    conn.commit()
                    
                    if symbol != '^NSEI':
                        info = yf.Ticker(ticker).info
                        sector = info.get('sector', 'Unknown')
                        cursor.execute("INSERT OR REPLACE INTO stock_metadata (Symbol, Sector) VALUES (?, ?)", (symbol, sector))
                        conn.commit()

                    tqdm.write(f"Successfully processed '{symbol}'")
                    time.sleep(0.5)
                    break 

                except Exception as e:
                    tqdm.write(f"--> FAILED '{symbol}' attempt {attempt+1}. Error: {e}")
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        tqdm.write(f"--> All retries failed for '{symbol}'.")

    print("\n--- Data ingestion process complete! ---")

if __name__ == '__main__':
    print("--- Initializing Databases (Will not delete existing data) ---")
    create_database()
    
    ingest_data()
    
    ingest_fama_french_data()