import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import importlib.util
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure comprehensive logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('portfolio_app_india.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_fetcher import DataFetcher
from utils.ai_analyzer import AIAnalyzer
from utils.database_manager import DatabaseManager
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.risk_calculator import RiskCalculator

def init_components():
    """Initialize application components with logging"""
    logger.info("Initializing India AI Portfolio Manager components...")
    
    try:
        if 'data_fetcher' not in st.session_state:
            logger.info("Creating new DataFetcher instance")
            st.session_state.data_fetcher = DataFetcher()
        
        if 'ai_analyzer' not in st.session_state:
            logger.info("Creating new AIAnalyzer instance")
            st.session_state.ai_analyzer = AIAnalyzer()
        
        if 'database_manager' not in st.session_state:
            logger.info("Creating new DatabaseManager instance")
            st.session_state.database_manager = DatabaseManager()
        
        if 'portfolio_optimizer' not in st.session_state:
            logger.info("Creating new PortfolioOptimizer instance")
            symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            st.session_state.portfolio_optimizer = PortfolioOptimizer(symbols=symbols)
        
        if 'risk_calculator' not in st.session_state:
            logger.info("Creating new RiskCalculator instance")
            symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            st.session_state.risk_calculator = RiskCalculator(symbols=symbols, market_index='^NSEI')
        
        logger.info("Component initialization complete")
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing application components: {str(e)}")

def main():
    logger.info("Starting India AI Portfolio Manager application")
    
    try:
        st.set_page_config(
            page_title="Indian AI Portfolio Manager",
            page_icon="🇮🇳",  # Indian flag emoji
            layout="wide",
            initial_sidebar_state="expanded"
        )
        logger.info("Page configuration set successfully")
    except Exception as e:
        logger.error(f"Error setting page config: {str(e)}")
        st.error(f"Error setting page configuration: {str(e)}")
        return

    # Custom navigation
    st.sidebar.markdown("<h1 style='font-size: 36px;'>🇮🇳 India AI Portfolio Manager</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("### Navigation")
    logger.info("Sidebar navigation initialized")

    # Navigation options tailored for Indian market
    page_selection = st.sidebar.selectbox(
        "Choose Section:",
        ["📈 Indian Market Analysis", "🧠 AI Portfolio Agent", "💾 Data Management"],
        index=0
    )
    logger.info(f"User selected page: {page_selection}")

    # Initialize components
    init_components()

    # Route to appropriate page based on selection
    try:
        if page_selection == "📈 Indian Market Analysis":
            logger.info("Loading Indian Market Analysis component")
            spec = importlib.util.spec_from_file_location("market_analysis", "components/Market_Analysis.py")
            if spec is None:
                raise ImportError("Market_Analysis.py not found")
            market_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(market_module)
            market_module.main()
            return
        elif page_selection == "🧠 AI Portfolio Agent":
            logger.info("Loading AI Portfolio Agent component")
            spec = importlib.util.spec_from_file_location("ai_agent", "components/AI_Agent.py")
            if spec is None:
                raise ImportError("AI_Agent.py not found")
            ai_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ai_module)
            ai_module.main()
            return
        elif page_selection == "💾 Data Management":
            logger.info("Loading Data Management component")
            spec = importlib.util.spec_from_file_location("data_management", "components/Data_Management.py")
            if spec is None:
                raise ImportError("Data_Management.py not found")
            data_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_module)
            data_module.main()
            return
    except Exception as e:
        logger.error(f"Error loading component {page_selection}: {str(e)}")
        st.error(f"Error loading {page_selection}. Please check logs for details.")
        return

    # Fallback section
    logger.warning("Reached fallback section - this should not happen")
    st.title("🇮🇳 India AI Portfolio Management System")
    st.markdown("### Please select a section from the navigation dropdown above")
    st.warning("No section loaded. Please select an option from the sidebar.")

if __name__ == "__main__":
    main()