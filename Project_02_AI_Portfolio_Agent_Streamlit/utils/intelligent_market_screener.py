import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import yfinance as yf
from .gemini_utils import NIFTY_SECTORS_QUERY_CONFIG, analyze_news_with_gemini

logger = logging.getLogger(__name__)

class IntelligentMarketScreener:
    """
    Advanced market screener for all stocks in NIFTY_SECTORS_QUERY_CONFIG, analyzing like a 50-year veteran trader.
    Uses yfinance for price data, Finnhub for fundamentals, and optionally Gemini for sentiment analysis.
    Incorporates comprehensive scoring (quality, growth, value, momentum) and sector allocation recommendations.
    """

    def __init__(self, finnhub_key: str = None, gemini_key: str = None, alpha_vantage_key: str = None):
        self.finnhub_key = finnhub_key
        self.gemini_key = gemini_key
        self.alpha_vantage_key = alpha_vantage_key # This key is unused in the current code but kept for structure
        self.name_to_ticker_map = {k.lower(): v for k, v in self._create_ticker_mapping().items()}
        # Extract all unique stocks and their sectors from NIFTY_SECTORS_QUERY_CONFIG
        self.all_stocks = []
        self.stock_to_sectors = {}
        for sector, data in NIFTY_SECTORS_QUERY_CONFIG.items():
            for stock_name in data['stocks'].keys():
                # Use the case-insensitive map
                ticker = self.name_to_ticker_map.get(stock_name.lower())
                
                if not ticker:
                    # If still not found, then use the fallback
                    ticker = self._convert_to_yfinance_ticker(stock_name)
                
                if ticker not in self.all_stocks:
                    self.all_stocks.append(ticker)
                
                if ticker not in self.stock_to_sectors:
                    self.stock_to_sectors[ticker] = []
                self.stock_to_sectors[ticker].append(sector)

        # Market cap thresholds (in INR)
        self.large_cap_threshold = 83000 * 10**7  # ~₹83,000 Cr
        self.mid_cap_threshold = 16600 * 10**7    # ~₹16,600 Cr

    def _create_ticker_mapping(self) -> Dict[str, str]:
        """Create a mapping of stock names to yfinance tickers."""
        # This mapping is comprehensive and correct as provided.
        return {
            "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCL Technologies": "HCLTECH.NS", "Wipro": "WIPRO.NS",
            "Tech Mahindra": "TECHM.NS", "LTIMindtree": "LTIM.NS", "Mphasis": "MPHASIS.NS",
            "Persistent Systems": "PERSISTENT.NS", "Coforge": "COFORGE.NS", "Zensar Technologies": "ZENSARTECH.NS",
            "Oracle Financial Services": "OFSS.NS", "Happiest Minds": "HAPPSTMNDS.NS", "Cyient": "CYIENT.NS",
            "Sonata Software": "SONATSOFTW.NS", "Intellect Design Arena": "INTELLECT.NS",
            "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS",
            "Axis Bank": "AXISBANK.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS", "IndusInd Bank": "INDUSINDBK.NS",
            "Bank of Baroda": "BANKBARODA.NS", "PNB": "PNB.NS", "Canara Bank": "CANBK.NS",
            "Yes Bank": "YESBANK.NS", "IDFC First Bank": "IDFCFIRSTB.NS", "Federal Bank": "FEDERALBNK.NS",
            "Bandhan Bank": "BANDHANBNK.NS", "Union Bank of India": "UNIONBANK.NS", "Indian Bank": "INDIANB.NS",
            "Maruti Suzuki": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS", "Mahindra & Mahindra": "M&M.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS", "Hero MotoCorp": "HEROMOTOCO.NS", "Eicher Motors": "EICHERMOT.NS",
            "TVS Motor": "TVSMOTOR.NS", "Ashok Leyland": "ASHOKLEY.NS", "Bharat Forge": "BHARATFORG.NS",
            "Bosch": "BOSCHLTD.NS", "MRF": "MRF.NS", "Apollo Tyres": "APOLLOTYRE.NS",
            "Exide Industries": "EXIDEIND.NS", "Balkrishna Industries": "BALKRISIND.NS", "Ceat": "CEATLTD.NS",
            "Sun Pharma": "SUNPHARMA.NS", "Dr Reddy's Labs": "DRREDDY.NS", "Cipla": "CIPLA.NS",
            "Divi's Laboratories": "DIVISLAB.NS", "Aurobindo Pharma": "AUROPHARMA.NS", "Lupin": "LUPIN.NS",
            "Torrent Pharma": "TORNTPHARM.NS", "Alkem Laboratories": "ALKEM.NS", "Zydus Lifesciences": "ZYDUSLIFE.NS",
            "Mankind Pharma": "MANKIND.NS", "Biocon": "BIOCON.NS", "Glenmark Pharmaceuticals": "GLENMARK.NS",
            "Laurus Labs": "LAURUSLABS.NS", "Ipca Laboratories": "IPCALAB.NS", "Abbott India": "ABBOTINDIA.NS",
            "Hindustan Unilever": "HINDUNILVR.NS", "ITC": "ITC.NS", "Nestle India": "NESTLEIND.NS",
            "Britannia Industries": "BRITANNIA.NS", "Dabur India": "DABUR.NS", "Godrej Consumer Products": "GODREJCP.NS",
            "Colgate-Palmolive India": "COLPAL.NS", "Marico": "MARICO.NS", "United Spirits": "MCDOWELL-N.NS",
            "Varun Beverages": "VBL.NS", "Emami": "EMAMILTD.NS", "Jyothy Labs": "JYOTHYLAB.NS",
            "Tata Consumer Products": "TATACONSUM.NS", "Patanjali Foods": "PATANJALI.NS", "Radico Khaitan": "RADICO.NS",
            "Havells India": "HAVELLS.NS", "Voltas": "VOLTAS.NS", "Whirlpool of India": "WHIRLPOOL.NS",
            "Crompton Greaves Consumer": "CROMPTON.NS", "Bajaj Electricals": "BAJAJELEC.NS", "Blue Star": "BLUESTARCO.NS",
            "V-Guard Industries": "VGUARD.NS", "Symphony": "SYMPHONY.NS", "TTK Prestige": "TTKPRESTIG.NS",
            "Orient Electric": "ORIENTELEC.NS", "Dixon Technologies": "DIXON.NS", "Amber Enterprises": "AMBER.NS",
            "Polycab India": "POLYCAB.NS", "Relaxo Footwears": "RELAXO.NS", "Century Plyboards": "CENTURYPLY.NS",
            "Bajaj Finance": "BAJFINANCE.NS", "HDFC Life Insurance": "HDFCLIFE.NS", "SBI Life Insurance": "SBILIFE.NS",
            "ICICI Prudential Life": "ICICIPRULI.NS", "Shriram Finance": "SHRIRAMFIN.NS", "Cholamandalam Investment": "CHOLAFIN.NS",
            "Power Finance Corporation": "PFC.NS", "REC Limited": "RECLTD.NS", "Muthoot Finance": "MUTHOOTFIN.NS",
            "Bajaj Finserv": "BAJAJFINSV.NS", "LIC Housing Finance": "LICHSGFIN.NS", "Aditya Birla Capital": "ABCAPITAL.NS",
            "Max Financial Services": "MFSL.NS", "Piramal Enterprises": "PEL.NS", "L&T Finance Holdings": "LTF.NS",
            "Zee Entertainment": "ZEEL.NS", "Sun TV Network": "SUNTV.NS", "PVR Inox": "PVRINOX.NS",
            "Dish TV India": "DISHTV.NS", "Network18 Media": "NETWORK18.NS", "TV Today Network": "TVTODAY.NS",
            "Jagran Prakashan": "JAGRAN.NS", "DB Corp": "DBCORP.NS", "Nazara Technologies": "NAZARA.NS",
            "Saregama India": "SAREGAMA.NS", "Hathway Cable": "HATHWAY.NS", "Tips Industries": "TIPSINDLTD.NS",
            "Balaji Telefilms": "BALAJITELE.NS", "Prime Focus": "PFOCUS.NS", "NDTV": "NDTV.NS",
            "Tata Steel": "TATASTEEL.NS", "JSW Steel": "JSWSTEEL.NS", "Hindalco Industries": "HINDALCO.NS",
            "Vedanta": "VEDL.NS", "SAIL": "SAIL.NS", "NALCO": "NATIONALUM.NS", "Jindal Steel & Power": "JINDALSTEL.NS",
            "APL Apollo Tubes": "APLAPOLLO.NS", "Ratnamani Metals": "RATNAMANI.NS", "Hindustan Zinc": "HINDZINC.NS",
            "NMDC": "NMDC.NS", "Welspun Corp": "WELCORP.NS", "JSL Stainless": "JSL.NS",
            "Hindustan Copper": "HINDCOPPER.NS", "MOIL": "MOIL.NS", "Bank of India": "BANKINDIA.NS",
            "Central Bank of India": "CENTRALBK.NS", "UCO Bank": "UCOBANK.NS", "Indian Overseas Bank": "IOB.NS",
            "Maharashtra Bank": "MAHABANK.NS", "Punjab & Sind Bank": "PSB.NS", "J&K Bank": "J&KBANK.NS",
            "IDBI Bank": "IDBI.NS", "RBL Bank": "RBLBANK.NS", "City Union Bank": "CUB.NS",
            "Karur Vysya Bank": "KARURVYSYA.NS", "DCB Bank": "DCBBANK.NS", "Equitas Small Finance Bank": "EQUITASBNK.NS",
            "AU Small Finance Bank": "AUBANK.NS", "DLF": "DLF.NS", "Godrej Properties": "GODREJPROP.NS",
            "Oberoi Realty": "OBEROIRLTY.NS", "Prestige Estates": "PRESTIGE.NS", "Brigade Enterprises": "BRIGADE.NS",
            "Sobha": "SOBHA.NS", "Phoenix Mills": "PHOENIXLTD.NS", "Macrotech Developers": "LODHA.NS",
            "Sunteck Realty": "SUNTECK.NS", "Mahindra Lifespace": "MAHLIFE.NS", "Kolte Patil Developers": "KOLTEPATIL.NS",
            "Puravankara": "PURVA.NS", "Anant Raj": "ANANTRAJ.NS", "Indiabulls Real Estate": "IBREALEST.NS",
            "NBCC India": "NBCC.NS", "UltraTech Cement": "ULTRACEMCO.NS", "Shree Cement": "SHREECEM.NS",
            "Asian Paints": "ASIANPAINT.NS", "Grasim Industries": "GRASIM.NS", "Pidilite Industries": "PIDILITIND.NS",
            "Ambuja Cements": "AMBUJACEM.NS", "ACC": "ACC.NS", "Tata Chemicals": "TATACHEM.NS",
            "UPL": "UPL.NS", "Coromandel International": "COROMANDEL.NS", "Deepak Nitrite": "DEEPAKNTR.NS",
            "SRF": "SRF.NS", "Gujarat Fluorochemicals": "FLUOROCHEM.NS", "Kansai Nerolac": "KANSAINER.NS",
            "Berger Paints": "BERGEPAINT.NS", "Reliance Industries": "RELIANCE.NS", "NTPC": "NTPC.NS",
            "Power Grid Corporation": "POWERGRID.NS", "Adani Green Energy": "ADANIGREEN.NS", "Tata Power": "TATAPOWER.NS",
            "Adani Power": "ADANIPOWER.NS", "JSW Energy": "JSWENERGY.NS", "NHPC": "NHPC.NS",
            "Torrent Power": "TORNTPOWER.NS", "SJVN": "SJVN.NS", "Indian Oil Corporation": "IOC.NS",
            "BPCL": "BPCL.NS", "GAIL India": "GAIL.NS", "Larsen & Toubro": "LT.NS",
            "Adani Ports and SEZ": "ADANIPORTS.NS", "GMR Airports Infrastructure": "GMRINFRA.NS",
            "IRB Infrastructure": "IRB.NS", "KNR Constructions": "KNRCON.NS", "PNC Infratech": "PNCINFRA.NS",
            "NCC": "NCC.NS", "Dilip Buildcon": "DBL.NS", "Ashoka Buildcon": "ASHOKA.NS",
            "GR Infraprojects": "GRINFRA.NS", "HG Infra Engineering": "HGINFRA.NS", "ONGC": "ONGC.NS",
            "Coal India": "COALINDIA.NS", "BHEL": "BHEL.NS", "HPCL": "HINDPETRO.NS",
            "Bharat Electronics": "BEL.NS", "Hindustan Aeronautics": "HAL.NS", "Oil India": "OIL.NS",
            "Container Corporation": "CONCOR.NS", "Cochin Shipyard": "COCHINSHIP.NS",
            "Mazagon Dock Shipbuilders": "MAZDOCK.NS", "Garden Reach Shipbuilders": "GRSE.NS",
            "RITES Ltd": "RITES.NS", "Engineers India Ltd": "ENGINERSIN.NS", "Petronet LNG": "PETRONET.NS",
            "Indraprastha Gas": "IGL.NS", "Mahanagar Gas": "MGL.NS", "Gujarat Gas": "GUJGASLTD.NS",
            "Adani Total Gas": "ATGL.NS", "Castrol India": "CASTROLIND.NS", "Gulf Oil Lubricants": "GULFOILLUB.NS",
            "Aegis Logistics": "AEGISCHEM.NS", "Apollo Hospitals": "APOLLOHOSP.NS", "Fortis Healthcare": "FORTIS.NS",
            "Max Healthcare": "MAXHEALTH.NS", "Metropolis Healthcare": "METROPOLIS.NS",
            "Dr Lal PathLabs": "LALPATHLAB.NS", "Narayana Hrudayalaya": "NH.NS", "Aster DM Healthcare": "ASTERDM.NS",
            "Thyrocare Technologies": "THYROCARE.NS", "Healthcare Global Enterprises": "HCG.NS",
            "Krishna Institute of Medical Sciences": "KIMS.NS", "Global Health Ltd (Medanta)": "MEDANTA.NS",
            "Rainbow Childrens Medicare": "RAINBOW.NS", "Kovai Medical Center": "KOVAI.NS",
            "Shalby Multi-specialty Hospitals": "SHALBY.NS", "Vijaya Diagnostic Centre": "VIJAYA.NS",
            "Vodafone Idea": "IDEA.NS", "Aarti Industries": "AARTIIND.NS", "Astral Ltd": "ASTRAL.NS",
            "Indian Hotels": "INDHOTEL.NS", "Bharti Airtel": "BHARTIARTL.NS", "Avenue Supermarts (DMart)": "DMART.NS",
            "Zomato": "ZOMATO.NS", "ITC Ltd": "ITC.NS", "Siemens India": "SIEMENS.NS",
            "Cummins India": "CUMMINSIND.NS", "ABB India": "ABB.NS"
        }

    def _convert_to_yfinance_ticker(self, stock_name: str) -> str:
        """Fallback method to convert stock name to yfinance ticker if not in mapping."""
        ticker = stock_name.replace(" ", "").replace("&", "").replace(".", "").upper() + ".NS"
        logger.warning(f"Using fallback ticker conversion for {stock_name}: {ticker}")
        return ticker

    def _get_finnhub_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data from Finnhub API."""
        if not self.finnhub_key:
            logger.error(f"No Finnhub API key provided for {ticker}")
            return None
        try:
            # Company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={self.finnhub_key}"
            profile_response = requests.get(profile_url, timeout=10)
            profile_response.raise_for_status()
            profile_data = profile_response.json()

            # Basic metrics
            metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={self.finnhub_key}"
            metrics_response = requests.get(metrics_url, timeout=10)
            metrics_response.raise_for_status()
            metrics_data = metrics_response.json()

            # Finnhub marketCap is in millions, so convert to absolute value
            market_cap_inr = profile_data.get('marketCapitalization', 0) * 1_000_000 * 83 # Assuming USD and 83 INR/USD
            
            fundamental_data = {
                'company_name': profile_data.get('name', ticker.replace('.NS', '')),
                'industry': profile_data.get('finnhubIndustry', 'Unknown'),
                'sector': profile_data.get('gicsSector', 'Unknown'),
                'market_cap': market_cap_inr,
                'country': profile_data.get('country', 'IN'),
                'ipo_date': profile_data.get('ipo', ''),
                'employee_count': profile_data.get('employeeTotal', 0),
                'pe_ratio': metrics_data.get('metric', {}).get('peBasicExclExtraTTM', np.nan),
                'pb_ratio': metrics_data.get('metric', {}).get('pbQuarterly', np.nan),
                'roe': metrics_data.get('metric', {}).get('roeRfy', np.nan),
                'roa': metrics_data.get('metric', {}).get('roaRfy', np.nan),
                'profit_margin': metrics_data.get('metric', {}).get('profitMarginRfy', np.nan),
                'debt_to_equity': metrics_data.get('metric', {}).get('totalDebt/totalEquityQuarterly', np.nan),
                'current_ratio': metrics_data.get('metric', {}).get('currentRatioQuarterly', np.nan),
                'revenue_growth': metrics_data.get('metric', {}).get('revenueGrowthRfy', np.nan),
                'eps_growth': metrics_data.get('metric', {}).get('epsGrowthRfy', np.nan),
                'beta': metrics_data.get('metric', {}).get('beta', np.nan),
                'eps': metrics_data.get('metric', {}).get('epsTTM', np.nan)
            }
            return fundamental_data
        except Exception as e:
            logger.error(f"Error fetching Finnhub data for {ticker}: {str(e)[:150]}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators like RSI, moving averages, and volatility."""
        try:
            if df.empty or len(df) < 2:
                raise ValueError("DataFrame is empty or too short for technical analysis.")

            # RSI (14-day)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Moving Averages
            sma50 = df['Close'].rolling(window=50).mean()
            sma200 = df['Close'].rolling(window=200).mean()

            # Volatility (30-day annualized)
            returns = df['Close'].pct_change().dropna()
            volatility = float(returns.tail(30).std() * np.sqrt(252)) if len(returns) >= 30 else np.nan

            # Momentum (30-day)
            price_30d = df['Close'].iloc[-30] if len(df) >= 30 else df['Close'].iloc[0]
            momentum_30d = ((df['Close'].iloc[-1] / price_30d) - 1) * 100 if price_30d > 0 else 0

            latest = df.iloc[-1]
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0,
                'sma50': sma50.iloc[-1] if not sma50.empty and not pd.isna(sma50.iloc[-1]) else latest['Close'],
                'sma200': sma200.iloc[-1] if not sma200.empty and not pd.isna(sma200.iloc[-1]) else latest['Close'],
                'price': latest['Close'],
                'volatility_annualized': volatility,
                'momentum_30d': momentum_30d,
                'avg_volume_30d': float(df['Volume'].tail(30).mean()) if len(df) >= 30 else df['Volume'].mean()
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)[:150]}")
            return {
                'rsi': 50.0, 'sma50': np.nan, 'sma200': np.nan, 'price': np.nan,
                'volatility_annualized': np.nan, 'momentum_30d': 0.0, 'avg_volume_30d': np.nan
            }

    def get_comprehensive_stock_data(self, ticker: str) -> Optional[Dict]:
        """Fetch comprehensive stock data using yfinance and Finnhub."""
        try:
            stock = yf.Ticker(ticker)
            # Use a more reliable way to get info that works for more tickers
            info = stock.fast_info
            hist = stock.history(period="1y")

            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return None

            fundamentals = self._get_finnhub_fundamentals(ticker) or {}
            technicals = self._calculate_technical_indicators(hist)

            # Prioritize yfinance info where available, then Finnhub
            market_cap_yf = info.get('market_cap', np.nan)
            market_cap_final = market_cap_yf if pd.notna(market_cap_yf) else fundamentals.get('market_cap', np.nan)

            data = {
                'ticker': ticker,
                'company_name': info.get('longName', fundamentals.get('company_name', ticker.replace('.NS', ''))),
                'sector': self.stock_to_sectors.get(ticker, [fundamentals.get('sector', 'Unknown')]),
                'market_cap': market_cap_final,
                'pe_ratio': fundamentals.get('pe_ratio', info.get('trailingPE', np.nan)),
                'pb_ratio': fundamentals.get('pb_ratio', info.get('priceToBook', np.nan)),
                'roe': fundamentals.get('roe', info.get('returnOnEquity', np.nan)),
                'roa': fundamentals.get('roa', info.get('returnOnAssets', np.nan)),
                'profit_margin': fundamentals.get('profit_margin', info.get('profitMargins', np.nan)),
                'debt_to_equity': fundamentals.get('debt_to_equity', info.get('debtToEquity', np.nan)),
                'current_ratio': fundamentals.get('current_ratio', info.get('currentRatio', np.nan)),
                'revenue_growth': fundamentals.get('revenue_growth', info.get('revenueGrowth', np.nan)),
                'eps_growth': fundamentals.get('eps_growth', info.get('earningsGrowth', np.nan)),
                'beta': fundamentals.get('beta', info.get('beta', np.nan)),
                'eps': fundamentals.get('eps', info.get('trailingEps', np.nan)),
                'current_price': technicals['price'],
                'rsi': technicals['rsi'],
                'sma50': technicals['sma50'],
                'sma200': technicals['sma200'],
                'volatility_annualized': technicals['volatility_annualized'],
                'momentum_30d': technicals['momentum_30d'],
                'avg_volume_30d': technicals['avg_volume_30d'],
                'price_data_available': True
            }
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)[:150]}")
            return None

    def _calculate_quality_score(self, stock: Dict) -> Optional[float]:
        """Calculate quality score based on financial health."""
        try:
            score = 5.0
            roe = stock.get('roe', np.nan)
            if not np.isnan(roe):
                # ROE in yfinance is a ratio, not percentage
                roe_pct = roe * 100
                if roe_pct > 20: score += 2
                elif roe_pct > 15: score += 1
                elif roe_pct < 5: score -= 1

            profit_margin = stock.get('profit_margin', np.nan)
            if not np.isnan(profit_margin):
                # profit_margin in yfinance is a ratio
                pm_pct = profit_margin * 100
                if pm_pct > 20: score += 1.5
                elif pm_pct > 10: score += 0.5
                elif pm_pct < 5: score -= 1

            debt_equity = stock.get('debt_to_equity', np.nan)
            if not np.isnan(debt_equity):
                # D/E in yfinance is already a value, not percentage
                if debt_equity < 30: score += 1 # In yfinance this is a percentage
                elif debt_equity > 100: score -= 1

            current_ratio = stock.get('current_ratio', np.nan)
            if not np.isnan(current_ratio):
                if current_ratio > 2: score += 0.5
                elif current_ratio < 1: score -= 1

            return max(0.0, min(10.0, score))
        except Exception as e:
            logger.warning(f"Error calculating quality score for {stock.get('ticker', 'Unknown')}: {str(e)[:150]}")
            return None

    def _calculate_growth_score(self, stock: Dict) -> Optional[float]:
        """Calculate growth potential score."""
        try:
            score = 5.0
            revenue_growth = stock.get('revenue_growth', np.nan)
            if not np.isnan(revenue_growth):
                # revenue_growth in yfinance is a ratio
                rg_pct = revenue_growth * 100
                if rg_pct > 20: score += 2
                elif rg_pct > 10: score += 1
                elif rg_pct < 0: score -= 2

            eps_growth = stock.get('eps_growth', np.nan)
            # yfinance calls this 'earningsGrowth'
            if 'earningsGrowth' in stock and not pd.isna(stock['earningsGrowth']):
                eps_growth = stock['earningsGrowth']

            if not np.isnan(eps_growth):
                eg_pct = eps_growth * 100
                if eg_pct > 15: score += 2
                elif eg_pct > 5: score += 1
                elif eg_pct < -10: score -= 2

            sectors = stock.get('sector', [])
            high_growth_sectors = ['Nifty IT', 'Nifty Healthcare', 'Nifty Financial Services', 'Nifty Consumer Durables']
            if any(sector in high_growth_sectors for sector in sectors):
                score += 0.5

            return max(0.0, min(10.0, score))
        except Exception as e:
            logger.warning(f"Error calculating growth score for {stock.get('ticker', 'Unknown')}: {str(e)[:150]}")
            return None

    def _calculate_value_score(self, stock: Dict) -> Optional[float]:
        """Calculate value investment score."""
        try:
            score = 5.0
            pe_ratio = stock.get('pe_ratio', np.nan)
            if not np.isnan(pe_ratio) and pe_ratio > 0:
                if pe_ratio < 15: score += 2
                elif pe_ratio < 25: score += 1
                elif pe_ratio > 40: score -= 1

            pb_ratio = stock.get('pb_ratio', np.nan)
            if not np.isnan(pb_ratio) and pb_ratio > 0:
                if pb_ratio < 2: score += 1
                elif pb_ratio > 5: score -= 1

            market_cap = stock.get('market_cap', np.nan)
            if not np.isnan(market_cap):
                if market_cap > self.large_cap_threshold:
                    score += 0.5

            return max(0.0, min(10.0, score))
        except Exception as e:
            logger.warning(f"Error calculating value score for {stock.get('ticker', 'Unknown')}: {str(e)[:150]}")
            return None

    def _calculate_momentum_score(self, stock: Dict) -> Optional[float]:
        """Calculate momentum and technical score."""
        try:
            score = 5.0
            momentum_30d = stock.get('momentum_30d', 0.0)
            if not np.isnan(momentum_30d):
                if momentum_30d > 10: score += 2
                elif momentum_30d > 5: score += 1
                elif momentum_30d < -10: score -= 2
                elif momentum_30d < -5: score -= 1

            volatility = stock.get('volatility_annualized', np.nan)
            if not np.isnan(volatility):
                if volatility < 0.20: score += 1
                elif volatility > 0.40: score -= 1

            avg_volume = stock.get('avg_volume_30d', np.nan)
            if not np.isnan(avg_volume) and avg_volume > 0:
                if avg_volume > 5_000_000: score += 0.5
                elif avg_volume < 500_000: score -= 0.5

            rsi = stock.get('rsi', 50.0)
            if not np.isnan(rsi):
                if 30 <= rsi <= 70: score += 0.2
                elif rsi < 30: score += 0.5 # Stronger signal for oversold
                elif rsi > 70: score -= 0.5 # Stronger penalty for overbought

            sma50 = stock.get('sma50', np.nan)
            sma200 = stock.get('sma200', np.nan)
            if not np.isnan(sma50) and not np.isnan(sma200):
                if sma50 > sma200:
                    score += 0.5 # Golden cross signal

            return max(0.0, min(10.0, score))
        except Exception as e:
            logger.warning(f"Error calculating momentum score for {stock.get('ticker', 'Unknown')}: {str(e)[:150]}")
            return None

    def _generate_investment_thesis(self, stock: Dict) -> str:
        """Generate investment thesis for the stock."""
        symbol = stock.get('ticker', '')
        company_name = stock.get('company_name', symbol)
        sectors = stock.get('sector', ['Unknown'])
        score = stock.get('investment_score', 0)

        if score >= 8:
            strength = "STRONG BUY"
            rationale = "exceptional fundamentals, strong growth prospects, and positive market momentum"
        elif score >= 7:
            strength = "BUY"
            rationale = "solid fundamentals with good upside potential and favorable technicals"
        elif score >= 6:
            strength = "ACCUMULATE"
            rationale = "decent fundamentals; a good candidate for accumulation on dips"
        elif score >= 5:
            strength = "HOLD"
            rationale = "fairly valued with mixed signals; suitable for existing positions"
        else:
            strength = "AVOID/REDUCE"
            rationale = "weak fundamentals, poor momentum, or appears overvalued"

        return f"{strength} | {company_name} ({', '.join(sectors)}) | Score: {score:.2f}/10. Thesis: {rationale}."

    def screen_best_opportunities(self, max_stocks: int = 50, min_score: float = 5.0, include_sentiment: bool = False) -> List[Dict]:
        """
        Screen the market for best investment opportunities among Indian stocks.
        Returns top opportunities based on comprehensive scoring.
        """
        logger.info(f"Screening {len(self.all_stocks)} Indian stocks for best opportunities...")
        all_stocks_data = []
        date_range_str = f"{(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}"

        for ticker in self.all_stocks:
            logger.info(f"Screening {ticker}")
            stock_data = self.get_comprehensive_stock_data(ticker)
            if stock_data and stock_data.get('price_data_available'):
                if include_sentiment and self.gemini_key:
                    # This part remains a placeholder as fetching real news is out of scope of this refactor
                    # In a real app, you would fetch articles here.
                    articles = [] 
                    sentiment_result, error = analyze_news_with_gemini(
                        _api_key=self.gemini_key, articles_texts_list=articles,
                        analysis_target_name=stock_data.get('company_name', ticker),
                        date_range_str=date_range_str, target_type="stock"
                    )
                    if sentiment_result and not error:
                        stock_data.update(sentiment_result)
                    else:
                        logger.warning(f"Sentiment analysis failed for {ticker}: {error}")
                        stock_data['sentiment'] = 'N/A'
                        stock_data['sentiment_score_llm'] = 0.0

                all_stocks_data.append(stock_data)

        logger.info(f"Successfully analyzed {len(all_stocks_data)} stocks")
        scored_stocks = self._apply_veteran_screening_criteria(all_stocks_data, include_sentiment)
        best_opportunities = sorted(scored_stocks, key=lambda x: x.get('investment_score', 0), reverse=True)
        return [stock for stock in best_opportunities if stock.get('investment_score', 0) >= min_score][:max_stocks]

    def _apply_veteran_screening_criteria(self, stocks_data: List[Dict], include_sentiment: bool) -> List[Dict]:
        """Apply comprehensive scoring criteria (quality, growth, value, momentum, sentiment)."""
        scored_stocks = []
        for stock in stocks_data:
            try:
                # Weighted average of the four scores
                weights = {'quality': 0.30, 'growth': 0.25, 'value': 0.20, 'momentum': 0.25}
                score = 0.0
                num_scores = 0
                
                quality_score = self._calculate_quality_score(stock)
                if quality_score is not None:
                    score += quality_score * weights['quality']
                    num_scores += 1
                
                growth_score = self._calculate_growth_score(stock)
                if growth_score is not None:
                    score += growth_score * weights['growth']
                    num_scores += 1

                value_score = self._calculate_value_score(stock)
                if value_score is not None:
                    score += value_score * weights['value']
                    num_scores += 1

                momentum_score = self._calculate_momentum_score(stock)
                if momentum_score is not None:
                    score += momentum_score * weights['momentum']
                    num_scores += 1

                if include_sentiment and 'sentiment_score_llm' in stock:
                    # Adding sentiment as a bonus/penalty, not a core weighted component
                    sentiment_bonus = stock.get('sentiment_score_llm', 0.0) * 0.5 # Max +/- 0.5 points
                    score += sentiment_bonus
                
                if num_scores > 0:
                    # Normalize the score
                    total_weight = sum(weights[k] for k in ['quality', 'growth', 'value', 'momentum'] if locals()[f'{k}_score'] is not None)
                    normalized_score = (score / total_weight) if total_weight > 0 else 0
                    stock['investment_score'] = round(max(0.0, min(10.0, normalized_score)), 2)
                    stock['composite_score'] = stock['investment_score'] # For compatibility
                    stock['investment_thesis'] = self._generate_investment_thesis(stock)
                    scored_stocks.append(stock)

            except Exception as e:
                logger.warning(f"Error scoring stock {stock.get('ticker', 'Unknown')}: {str(e)[:150]}")
                continue

        return scored_stocks

    def get_sector_allocation_recommendations(self, screened_stocks: List[Dict]) -> Dict[str, float]:
        """Get sector allocation recommendations based on the strength of top stocks in each sector."""
        if not screened_stocks:
            return {}

        sector_scores = {}
        for stock in screened_stocks:
            score = stock.get('investment_score', 0)
            # Use the primary sector from the config
            primary_sector = self.stock_to_sectors.get(stock['ticker'], ['Unknown'])[0]
            if primary_sector not in sector_scores:
                sector_scores[primary_sector] = []
            sector_scores[primary_sector].append(score)

        sector_strength = {}
        for sector, scores in sector_scores.items():
            # Strength is a combination of average score and number of high-quality stocks
            avg_score = np.mean(scores)
            num_strong_stocks = len([s for s in scores if s > 7.0])
            strength = avg_score * (1 + 0.1 * num_strong_stocks) # Bonus for more strong stocks
            sector_strength[sector] = strength
        
        total_strength = sum(sector_strength.values())
        if total_strength == 0:
            return {}

        sector_allocations = {sector: (strength / total_strength) * 100 for sector, strength in sector_strength.items()}
        return dict(sorted(sector_allocations.items(), key=lambda x: x[1], reverse=True))


    def get_sector_summary(self) -> Dict[str, Dict]:
        """Summarize performance by sector by fetching and scoring a small sample of stocks."""
        # This is a simplified version for quick summary. The main screening is more thorough.
        sector_summary = {}
        for sector, data in NIFTY_SECTORS_QUERY_CONFIG.items():
            top_stocks_in_sector = list(data['stocks'].keys())[:3] # Sample 3 stocks
            scores = []
            stock_details = []
            for stock_name in top_stocks_in_sector:
                ticker = self.ticker_mapping.get(stock_name)
                if not ticker: continue
                
                stock_data = self.get_comprehensive_stock_data(ticker)
                if stock_data:
                    score = self._apply_veteran_screening_criteria([stock_data], False)
                    if score:
                        scores.append(score[0]['investment_score'])
                        stock_details.append({'ticker': ticker, 'score': score[0]['investment_score']})
            
            if scores:
                sector_summary[sector] = {
                    'avg_score': round(np.mean(scores), 2),
                    'top_stocks_sample': sorted(stock_details, key=lambda x:x['score'], reverse=True),
                    'count': len(scores)
                }

        return sector_summary