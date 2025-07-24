# utils/ticker_universe.py

"""
Central repository for all Indian stock and index tickers.
This module provides a comprehensive mapping of company names to their yfinance tickers
and organizes them by sector based on the NIFTY indices.
"""

# The configuration you provided, expanded for clarity and use across the app.
NIFTY_SECTORS_QUERY_CONFIG = {
    "Nifty 50": {"ticker": "^NSEI", "stocks": {}}, # Added for completeness
    "Sensex": {"ticker": "^BSESN", "stocks": {}}, # Added for completeness
    "Nifty Bank": {"ticker": "^NSEBANK", "stocks": {
        "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS", "Axis Bank": "AXISBANK.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS", "IndusInd Bank": "INDUSINDBK.NS", "Bank of Baroda": "BANKBARODA.NS",
        "PNB": "PNB.NS", "Canara Bank": "CANBK.NS", "IDFC First Bank": "IDFCFIRSTB.NS", "Federal Bank": "FEDERALBNK.NS",
        "Bandhan Bank": "BANDHANBNK.NS", "Union Bank of India": "UNIONBANK.NS", "Indian Bank": "INDIANB.NS"
    }},
    "Nifty IT": {"ticker": "^CNXIT", "stocks": {
        "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCL Technologies": "HCLTECH.NS", "Wipro": "WIPRO.NS",
        "Tech Mahindra": "TECHM.NS", "LTIMindtree": "LTIM.NS", "Mphasis": "MPHASIS.NS",
        "Persistent Systems": "PERSISTENT.NS", "Coforge": "COFORGE.NS", "Oracle Financial Services": "OFSS.NS"
    }},
    "Nifty Auto": {"ticker": "^CNXAUTO", "stocks": {
        "Maruti Suzuki": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS", "Mahindra & Mahindra": "M&M.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS", "Hero MotoCorp": "HEROMOTOCO.NS", "Eicher Motors": "EICHERMOT.NS",
        "TVS Motor": "TVSMOTOR.NS", "Ashok Leyland": "ASHOKLEY.NS", "Bharat Forge": "BHARATFORG.NS", "MRF": "MRF.NS"
    }},
    "Nifty Pharma": {"ticker": "^CNXPHARMA", "stocks": {
        "Sun Pharma": "SUNPHARMA.NS", "Dr Reddy's Labs": "DRREDDY.NS", "Cipla": "CIPLA.NS",
        "Divi's Laboratories": "DIVISLAB.NS", "Aurobindo Pharma": "AUROPHARMA.NS", "Lupin": "LUPIN.NS",
        "Zydus Lifesciences": "ZYDUSLIFE.NS", "Mankind Pharma": "MANKIND.NS", "Torrent Pharma": "TORNTPHARM.NS"
    }},
    "Nifty FMCG": {"ticker": "^CNXFMCG", "stocks": {
        "Hindustan Unilever": "HINDUNILVR.NS", "ITC": "ITC.NS", "Nestle India": "NESTLEIND.NS",
        "Britannia Industries": "BRITANNIA.NS", "Dabur India": "DABUR.NS", "Tata Consumer Products": "TATACONSUM.NS",
        "Godrej Consumer Products": "GODREJCP.NS", "Marico": "MARICO.NS", "Colgate-Palmolive India": "COLPAL.NS"
    }},
    # Add other sectors here if needed, following the same structure
}

def get_full_ticker_map() -> dict:
    """Returns a single dictionary mapping all stock names to their tickers."""
    full_map = {}
    for sector_info in NIFTY_SECTORS_QUERY_CONFIG.values():
        full_map.update(sector_info['stocks'])
    return full_map

def get_all_stock_tickers() -> list:
    """Returns a flat list of all unique stock tickers."""
    all_tickers = set()
    for sector_info in NIFTY_SECTORS_QUERY_CONFIG.values():
        for ticker in sector_info['stocks'].values():
            all_tickers.add(ticker)
    return sorted(list(all_tickers))

def get_sector_index_map() -> dict:
    """Returns a map of sector names to their index tickers."""
    return {
        sector: info["ticker"]
        for sector, info in NIFTY_SECTORS_QUERY_CONFIG.items()
        if "ticker" in info and info["ticker"]
    }

# Example of how to add more stocks programmatically in the future
# For now, we use the comprehensive config above.
# def load_tickers_from_csv(filepath):
#     import pandas as pd
#     df = pd.read_csv(filepath)
#     return dict(zip(df['COMPANY_NAME'], df['TICKER']))