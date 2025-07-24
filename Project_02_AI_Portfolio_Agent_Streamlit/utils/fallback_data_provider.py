import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from .database_manager import DatabaseManager

class FallbackDataProvider:
    """Provides fallback market data for Indian stocks when APIs are unavailable (for testing purposes)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        
        # Base prices for Indian stocks and indices in INR (indicative, based on 2025 market levels)
        self.base_prices = {
            # Major Indian stocks
            'RELIANCE.NS': 3000.0,    # Reliance Industries
            'TCS.NS': 4000.0,         # Tata Consultancy Services
            'HDFCBANK.NS': 1600.0,    # HDFC Bank
            'INFY.NS': 1800.0,        # Infosys
            'ICICIBANK.NS': 1200.0,   # ICICI Bank
            'HINDUNILVR.NS': 2700.0,  # Hindustan Unilever
            'DABUR.NS': 600.0,        # Dabur India
            'UPL.NS': 550.0,          # UPL Limited
            
            # Corrected Sector indices
            '^CNXIT': 40000.0,
            '^NSEBANK': 50000.0,
            '^CNXFMCG': 60000.0,
            '^CNXAUTO': 25000.0,
            '^CNXPHARMA': 20000.0,
            
            # Market index
            '^NSEI': 24000.0,          # Nifty 50 Index
            '^BSESN': 82000.0 
        }
        
        self.volatility_map = {
            # Higher volatility for mid-caps, lower for large-caps and indices
            'RELIANCE.NS': 0.015, 'TCS.NS': 0.012, 'HDFCBANK.NS': 0.013, 'INFY.NS': 0.014, 
            'ICICIBANK.NS': 0.015, 'HINDUNILVR.NS': 0.012, 'DABUR.NS': 0.018, 'UPL.NS': 0.020,
            '^CNXIT': 0.010, '^NSEBANK': 0.011, '^CNXFMCG': 0.009, 
            '^CNXAUTO': 0.013, '^CNXPHARMA': 0.014, '^NSEI': 0.008, '^BSESN': 0.008
        }
        
        self.volume_map = {
            # Approximate daily volumes (in shares for stocks, units for indices)
            'RELIANCE.NS': 6000000, 'TCS.NS': 2000000, 'HDFCBANK.NS': 8000000, 'INFY.NS': 5000000,
            'ICICIBANK.NS': 10000000, 'HINDUNILVR.NS': 2000000, 'DABUR.NS': 1500000, 'UPL.NS': 2500000,
           '^CNXIT': 100000, '^NSEBANK': 150000, '^CNXFMCG': 80000, 
            '^CNXAUTO': 90000, '^CNXPHARMA': 120000, '^NSEI': 200000,  '^BSESN': 150000
        }
    
    def generate_stock_data(self, symbol, days=100):
        """Generate realistic synthetic stock data for Indian stocks in INR"""
        if symbol not in self.base_prices:
            self.logger.warning(f"Symbol {symbol} not supported in fallback data")
            return pd.DataFrame()
        
        try:
            base_price = self.base_prices[symbol]
            volatility = self.volatility_map.get(symbol, 0.015)  # Default volatility
            base_volume = self.volume_map.get(symbol, 1000000)   # Default volume
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Generate realistic price movements
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            returns = np.random.normal(0.0003, volatility, days)  # Small daily returns
            
            prices = [base_price]
            for i in range(1, days):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 1.0))  # Ensure positive prices
            
            # Generate OHLCV data
            data = []
            for i, date in enumerate(dates):
                close = prices[i]
                daily_vol = abs(np.random.normal(0, volatility / 2))  # Intraday volatility
                
                high = close * (1 + daily_vol)
                low = close * (1 - daily_vol)
                open_price = close * (1 + np.random.normal(0, volatility / 4))
                
                # Ensure logical OHLC order
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                volume = int(np.random.normal(base_volume, base_volume * 0.3))
                volume = max(volume, base_volume // 10)  # Minimum volume
                
                data.append({
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            
            # --- START OF FIX ---
            # This line is critical and must be present.
            df['source'] = 'synthetic'
            # --- END OF FIX ---
            
            # Store in DatabaseManager with synthetic data indicator
            if self.db_manager.is_available():
                df_for_db = df.copy()
                df_for_db['timestamp'] = df_for_db.index
                df_for_db['additional_data'] = json.dumps({'source': 'synthetic', 'generated_at': datetime.now().isoformat()})
                self.db_manager.store_market_data(symbol, 'stock', df_for_db)
                self.logger.info(f"Stored synthetic data for {symbol} in database")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error generating stock data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_summary(self):
        """Get a summary of major Indian market movements"""
        try:
            summary = {}
            key_indices = ['^NSEI', 'BANKNIFTY.NS', 'NIFTYIT.NS', 'NIFTYFMCG.NS']
            
            for symbol in key_indices:
                data = self.generate_stock_data(symbol, 30)
                if not data.empty:
                    current_price = data['close'].iloc[-1]
                    prev_price = data['close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    summary[symbol] = {
                        'price': round(current_price, 2),
                        'change_pct': round(change_pct, 2),
                        'status': 'synthetic_data',
                        'timestamp': datetime.now().isoformat()
                    }
            
            if self.db_manager.is_available():
                for symbol, info in summary.items():
                    df = pd.DataFrame({
                        'open': [info['price']],
                        'high': [info['price']],
                        'low': [info['price']],
                        'close': [info['price']],
                        'volume': [self.volume_map.get(symbol, 100000)],
                        'timestamp': [info['timestamp']],
                        'additional_data': [json.dumps({'source': 'synthetic', 'change_pct': info['change_pct']})]
                    })
                    self.db_manager.store_market_data(symbol, 'stock', df)
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return {}