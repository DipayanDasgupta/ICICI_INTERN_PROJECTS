# app/config.py

DB_FILE = "market_data.db"
PORTFOLIOS_DB_FILE = "user_portfolios.db" 

# --- UPDATED STOCK_UNIVERSES DICTIONARY ---
STOCK_UNIVERSES = {
    "NIFTY_50": [
        'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
        'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
        'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR',
        'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LTIM', 'LT', 'M&M',
        'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
        'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'WIPRO'
    ],
    "NIFTY_NEXT_50": [
        'ACC', 'ADANIENSOL', 'ADANIGREEN', 'AMBUJACEM', 'DMART', 'BAJAJHLDNG', 'BANKBARODA', 'BERGEPAINT', 'BEL',
        'BOSCHLTD', 'CHOLAFIN', 'COLPAL', 'DLF', 'DABUR', 'GAIL', 'GODREJCP', 'HAVELLS', 'HAL', 'ICICIGI',
        'ICICIPRULI', 'IOC', 'IGL', 'INDIGO', 'JSWENERGY', 'LICI', 'MARICO', 'MOTHERSON', 'MUTHOOTFIN',
        'NAUKRI', 'PIDILITIND', 'PEL', 'PNB', 'PGHH', 'SIEMENS', 'SBICARD', 'SHREECEM', 'SRF',
        'TATAPOWER', 'TVSMOTOR', 'TRENT', 'VEDL', 'VBL', 'ZEEL', 'ZOMATO'
    ],
    # Combined list for convenience in the Portfolio Studio
    "NIFTY_100_COMBINED": sorted(list(set([
        'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
        'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
        'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR',
        'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LTIM', 'LT', 'M&M',
        'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
        'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'WIPRO'
    ] + [
        'ACC', 'ADANIENSOL', 'ADANIGREEN', 'AMBUJACEM', 'DMART', 'BAJAJHLDNG', 'BANKBARODA', 'BERGEPAINT', 'BEL',
        'BOSCHLTD', 'CHOLAFIN', 'COLPAL', 'DLF', 'DABUR', 'GAIL', 'GODREJCP', 'HAVELLS', 'HAL', 'ICICIGI',
        'ICICIPRULI', 'IOC', 'IGL', 'INDIGO', 'JSWENERGY', 'LICI', 'MARICO', 'MOTHERSON', 'MUTHOOTFIN',
        'NAUKRI', 'PIDILITIND', 'PEL', 'PNB', 'PGHH', 'SIEMENS', 'SBICARD', 'SHREECEM', 'SRF',
        'TATAPOWER', 'TVSMOTOR', 'TRENT', 'VEDL', 'VBL', 'ZEEL', 'ZOMATO'
    ]))),
    "NIFTY_500": [
        '360ONE', 'AARTIDRUGS', 'AARTIIND', 'AAVAS', 'ABBOTINDIA', 'ABCAPITAL', 'ABFRL', 'ACC', 'ACE',
        'ADANIENSOL', 'ADANIENT', 'ADANIGREEN', 'ADANIPORTS', 'ADANIPOWER', 'ATGL', 'AWL', 'AEGISCHEM',
        'AETHER', 'AFFLE', 'AIAENG', 'AJANTPHARM', 'APLLTD', 'ALKEM', 'ALLCARGO', 'AMARAJABAT',
        'AMBER', 'AMBUJACEM', 'ANGELONE', 'ANURAS', 'APLAPOLLO', 'APOLLOHOSP', 'APOLLOTYRE', 'APTUS',
        'ASAHIINDIA', 'ASHOKLEY', 'ASIANPAINT', 'ASTERDM', 'ASTRAZEN', 'ASTRAL', 'AUROPHARMA',
        'AUSMALLFIN', 'AVANTIFEED', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJELEC', 'BAJFINANCE', 'BAJAJFINSV',
        'BAJAJHLDNG', 'BALAMINES', 'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA',
        'MAHABANK', 'BATAINDIA', 'BDL', 'BEL', 'BEML', 'BERGEPAINT', 'BHARATFORG', 'BHARTIARTL',
        'BHEL', 'BIOCON', 'BIRLACORPN', 'BSOFT', 'BLUEDART', 'BLUESTARCO', 'BOSCHLTD', 'BPCL',
        'BRIGADE', 'BRITANNIA', 'BROFINS', 'CANBK', 'CANFINHOME', 'CAPLIPOINT', 'CGCL', 'CARBORUNIV',
        'CASTROLIND', 'CCL', 'CDSL', 'CEATLTD', 'CENTRALBK', 'CENTURYPLY', 'CENTURYTEX', 'CERA',
        'CESC', 'CGPOWER', 'CHALET', 'CHAMBLFERT', 'CHENNPETRO', 'CHOLAFIN', 'CHOLAHLDNG', 'CIPLA',
        'CUB', 'CLEAN', 'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COLPAL', 'CONCOR', 'COROMANDEL',
        'CREDITACC', 'CRISIL', 'CROMPTON', 'CUMMINSIND', 'CYIENT', 'DAAWAT', 'DABUR', 'DALBHARAT',
        'DATAPATTNS', 'DCMSHRIRAM', 'DEEPAKNTR', 'DELHIVERY', 'DELTACORP', 'DEVYANI', 'DHANI',
        'DBL', 'DIVISLAB', 'DIXON', 'DLF', 'DMART', 'DOLLAR', 'DRREDDY', 'ECLERX', 'EDELWEISS',
        'EIDPARRY', 'EICHERMOT', 'EIHOTEL', 'ELGIEQUIP', 'EMAMILTD', 'ENDURANCE', 'ENGINERSIN',
        'EQUITASBNK', 'ERIS', 'ESCORTS', 'EXIDEIND', 'FDC', 'FACT', 'FEDERALBNK', 'FIVESTAR',
        'FINCABLES', 'FINPIPE', 'FINOLEXIND', 'FLUOROCHEM', 'FORTIS', 'FSL', 'GAIL', 'GAEL',
        'GATEWAY', 'GENSOL', 'GESHIP', 'GET&D', 'GILLETTE', 'GLAND', 'GLENMARK', 'GMDCLTD', 'GMRINFRA',
        'GNFC', 'GOCOLORS', 'GODFRYPHLP', 'GODREJAGRO', 'GODREJCP', 'GODREJIND', 'GODREJPROP',
        'GPPL', 'GRANULES', 'GRAPHITE', 'GRASIM', 'GRAVITA', 'GRINDWELL', 'GRINFRA', 'GSFC', 'GSPL',
        'GTPL', 'GUJALKALI', 'GUJGASLTD', 'GULFOILLUB', 'HAL', 'HAPPSTMNDS', 'HAVELLS', 'HBLPOWER',
        'HCLTECH', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HEG', 'HEMIPROP', 'HEROMOTOCO', 'HFCL',
        'HIKAL', 'HINDALCO', 'HINDCOPPER', 'HINDCOMPOS', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC',
        'POWERINDIA', 'HOMEFIRST', 'HONAUT', 'HUDCO', 'IBULHSGFIN', 'IEX', 'IFBIND', 'IFCI',
        'INDIACEM', 'INDIAMART', 'INDIANB', 'IOB', 'IOB', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI',
        'IDBI', 'IDFC', 'IDFCFIRSTB', 'IFBIND', 'IGL', 'IIFL', 'IIFLWAM', 'INDHOTEL', 'INDIANB',
        'INDIGO', 'INDOCO', 'INDUSINDBK', 'INDUSTOWER', 'INEOSSTYRO', 'INFIBEAM', 'INFY', 'INGERRAND',
        'INTELLECT', 'IOC', 'IPCALAB', 'IRB', 'IRCON', 'IRCTC', 'IRFC', 'ISEC', 'ITC', 'ITI', 'J&KBANK',
        'JAYNECOIND', 'JBCHEPHARM', 'JBMAUTO', 'JINDALSAW', 'JINDALSTEL', 'JKCEMENT', 'JKIL', 'JKLAKSHMI',
        'JKPAPER', 'JMFINANCIL', 'JPPOWER', 'JSL', 'JSWENERGY', 'JSWSTEEL', 'JUBILANT', 'JUBLFOOD',
        'JUBLINGT', 'JUSTDIAL', 'JYOTHYLAB', 'KABRAEXTRU', 'KAJARIACER', 'KALPATPOWR', 'KALYANKJIL',
        'KANSAINER', 'KARURVYSYA', 'KEC', 'KEI', 'KFINTECH', 'KHADIM', 'KIMS', 'KIRLOSENG',
        'KPRMILL', 'KRBL', 'KSB', 'KSL', 'KOTAKBANK', 'KPITTECH', 'L&TFH', 'LALPATHLAB', 'LAOPALA',
        'LAURUSLABS', 'LAXMIMACH', 'LEMONTREE', 'LICHSGFIN', 'LICI', 'LINDEINDIA', 'LT', 'LTIM',
        'LTTS', 'LUXIND', 'LXCHEM', 'M&M', 'M&MFIN', 'MANAPPURAM', 'MRPL', 'MAPMYINDIA', 'MARICO',
        'MARUTI', 'MASTEK', 'MAXESTATES', 'MAXHEALTH', 'MAZDOCK', 'MEDANTA', 'METROBRAND', 'METROPOLIS',
        'MFSL', 'MGL', 'MOTHERSON', 'MOTILALOFS', 'MPHASIS', 'MRF', 'MRPL', 'MSUMI', 'MTARTECH',
        'MUTHOOTFIN', 'NATIONALUM', 'NBCC', 'NCC', 'NAM-INDIA', 'NAUKRI', 'NAVINFLUOR', 'NAZARA',
        'NESTLEIND', 'NETWORK18', 'NEWGEN', 'NH', 'NHPC', 'NLCINDIA', 'NMDC', 'NOCIL', 'NTPC',
        'NYKAA', 'OAL', 'OBEROIRLTY', 'OFSS', 'ONGC', 'ORIENTELEC', 'PAGEIND', 'PAISALO', 'PATANJALI',
        'PATELENG', 'PAYTM', 'PCBL', 'PFC', 'PEL', 'PERSISTENT', 'PETRONET', 'PFIZER', 'PGHH', 'PGHL',
        'PNBHOUSING', 'PNCINFRA', 'PNB', 'POKARNA', 'POLYMED', 'POLYCAB', 'POONAWALLA', 'POWERGRID',
        'PPLPHARMA', 'PRESTIGE', 'PRINCEPIPE', 'PRSMJOHNSN', 'PRIVISCL', 'PSB', 'PTC', 'PVRINOX',
        'QUESS', 'RADHIKAJ', 'RAILTEL', 'RVNL', 'RAJESHEXPO', 'RALLIS', 'RCF', 'RTNINDIA',
        'RATNAMANI', 'RAYMOND', 'RBA', 'RECLTD', 'REDINGTON', 'RELAXO', 'RELIANCE', 'RENUKA',
        'RHIM', 'RITES', 'RKFORGE', 'ROLEXRINGS', 'ROUTE', 'RRKABEL', 'RSYSTEMS', 'SAIL', 'SANOFI',
        'SAPPHIRE', 'SAREGAMA', 'SCHAEFFLER', 'SFL', 'SHARDACROP', 'SHOPERSTOP', 'SHREECEM',
        'SHRIPISTON', 'SHRIRAMFIN', 'SIEMENS', 'SJVN', 'SKFINDIA', 'SOBHA', 'SOLARINDS', 'SONACOMS',
        'SONATSOFTW', 'SOUTHBANK', 'SPANDANA', 'SPARC', 'SRF', 'STARHEALTH', 'SBIN', 'SAIL',
        'SWSOLAR', 'SUDARSCHEM', 'SUMICHEM', 'SUNDARMFIN', 'SUNDRMFAST', 'SUNPHARMA', 'SUNTV',
        'SUPRAJIT', 'SUPREMEIND', 'SURANASOL', 'SURYODAY', 'SUZLON', 'SYMPHONY', 'SYNGENE', 'TASTYBITE',
        'TATACHEM', 'TATACOMM', 'TATACONSUM', 'TATAELXSI', 'TATAINVEST', 'TATAMTRDVR', 'TATAMOTORS',
        'TATAPOWER', 'TATASTEEL', 'TTML', 'TCS', 'TEAMLEASE', 'TECHM', 'TEJASNET', 'THERMAX',
        'THOMASCOOK', 'TIMKEN', 'TIPSINDLTD', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TRENT', 'TRIDENT',
        'TRIVENI', 'TRITURBINE', 'TTKPRESTIG', 'TV18BRDCST', 'TVSMOTOR', 'TVSSCS', 'UBL', 'UCOBANK',
        'UFLEX', 'UJJIVANSFB', 'ULTRACEMCO', 'UNICHEMLAB', 'UNIONBANK', 'UPL', 'UTIAMC', 'VAIBHAVGBL',
        'VAKRANGEE', 'VARROC', 'VBL', 'VEDL', 'VENKEYS', 'VIJAYA', 'VOLTAS', 'VRLLOG', 'WELCORP',
        'WELSPUNIND', 'WESTLIFE', 'WHIRLPOOL', 'WIPRO', 'WOCKPHARMA', 'YESBANK', 'ZEEL', 'ZENSARTECH',
        'ZFCVINDIA', 'ZOMATO', 'ZYDUSLIFE', 'ZYDUSWELL'
    ]
}