import streamlit as st
import sys
import os
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.autonomous_agent import AutonomousPortfolioAgent
from utils.data_fetcher import DataFetcher
from utils.database_manager import DatabaseManager
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.risk_calculator import RiskCalculator
from utils.ai_analyzer import AIAnalyzer
from utils.fallback_data_provider import FallbackDataProvider

# Set up logging for this module
logger = logging.getLogger(__name__)

def init_components():
    """Initialize all required components with logging"""
    logger.info("Initializing India AI Portfolio Agent components...")
    
    try:
        if 'autonomous_agent' not in st.session_state:
            logger.info("Creating AutonomousPortfolioAgent")
            st.session_state.autonomous_agent = AutonomousPortfolioAgent()
        
        if 'data_fetcher' not in st.session_state:
            logger.info("Creating DataFetcher for AI Agent")
            st.session_state.data_fetcher = DataFetcher()
        
        if 'db_manager' not in st.session_state:
            logger.info("Creating DatabaseManager for AI Agent")
            st.session_state.db_manager = DatabaseManager()
        
        if 'portfolio_optimizer' not in st.session_state:
            logger.info("Creating PortfolioOptimizer")
            symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            st.session_state.portfolio_optimizer = PortfolioOptimizer(symbols=symbols)
        
        if 'risk_calculator' not in st.session_state:
            logger.info("Creating RiskCalculator")
            symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            st.session_state.risk_calculator = RiskCalculator(symbols=symbols, market_index='^NSEI')
        
        if 'fallback_data_provider' not in st.session_state:
            logger.info("Creating FallbackDataProvider")
            st.session_state.fallback_data_provider = FallbackDataProvider()
        
        logger.info("AI Agent components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing AI Agent components: {str(e)}")
        st.error(f"Initialization error: {str(e)}")

def main():
    logger.info("Starting India AI Portfolio Agent main function")
    
    try:
        # Initialize components
        init_components()
        
        agent = st.session_state.autonomous_agent
        data_fetcher = st.session_state.data_fetcher
        db = st.session_state.db_manager
        optimizer = st.session_state.portfolio_optimizer
        risk_calc = st.session_state.risk_calculator
        fallback = st.session_state.fallback_data_provider
        
        logger.info("All components loaded successfully for AI Agent")
    except Exception as e:
        logger.error(f"Error initializing AI Agent components: {str(e)}")
        st.error(f"Initialization error: {str(e)}")
        return
    
    st.title("🇮🇳 India AI Portfolio Agent")
    st.markdown("### Your AI financial advisor for the Indian market")
    
    # Action selection with buttons
    st.subheader("🎯 What would you like the AI Agent to do?")
    
    # Initialize session state
    if 'selected_action' not in st.session_state:
        st.session_state.selected_action = "create_portfolio"
    
    # Action options mapping
    action_options = {
        "💰 Create Optimal Portfolio": "create_portfolio",
        "📊 Conduct Market Research": "market_research",
        "🎲 Run Trading Simulation": "trading_simulation",
        "🔮 Predict Market Dynamics": "market_prediction",
        "📈 Full Portfolio Analysis": "portfolio_analysis"
    }
    
    # Create two-column layout for buttons
    col1, col2 = st.columns(2)
    
    # Split options into two columns
    option_list = list(action_options.items())
    left_options = option_list[:3]  # First 3 options
    right_options = option_list[3:]  # Last 2 options
    
    with col1:
        for display_name, action_key in left_options:
            if st.button(display_name, 
                        type="primary" if st.session_state.selected_action == action_key else "secondary",
                        use_container_width=True,
                        key=f"btn_{action_key}"):
                st.session_state.selected_action = action_key
                st.rerun()
    
    with col2:
        for display_name, action_key in right_options:
            if st.button(display_name, 
                        type="primary" if st.session_state.selected_action == action_key else "secondary",
                        use_container_width=True,
                        key=f"btn_{action_key}"):
                st.session_state.selected_action = action_key
                st.rerun()
    
    st.markdown("---")
    
    # Handle selected actions
    if 'selected_action' in st.session_state:
        action = st.session_state.selected_action
        
        if action == "create_portfolio":
            st.subheader("💰 Create Your Optimal Portfolio")
            
            col_input1, col_input2 = st.columns(2)
            
            with col_input1:
                investment_amount = st.number_input(
                    "Investment Amount (₹)",
                    min_value=100000.0,
                    max_value=100000000.0,
                    value=5000000.0,
                    step=100000.0
                )
                
                risk_profile = st.selectbox(
                    "Risk Profile",
                    options=['conservative', 'moderate', 'aggressive'],
                    index=1
                )
            
            with col_input2:
                st.markdown("**Sectors to Include:**")
                include_equity = st.checkbox("📈 Equity", value=True)
                include_sectors = st.multiselect(
                    "Select Sectors",
                    options=['Nifty IT', 'Nifty Bank', 'Nifty FMCG', 'Nifty Auto', 'Nifty Pharma'],
                    default=['Nifty IT', 'Nifty Bank']
                )
            
            # Build sectors list
            sectors = []
            if include_equity:
                sectors.append('equity')
                sectors.extend([s.lower().replace(' ', '_') for s in include_sectors])
            
            if st.button("🚀 Generate Portfolio", type="primary"):
                if sectors:
                    logger.info(f"User requesting portfolio creation: ₹{investment_amount}, {risk_profile}, {sectors}")
                    with st.spinner("Analyzing Indian stocks and creating your portfolio..."):
                        try:
                            # Fetch data for optimization
                            symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
                            returns_data = pd.DataFrame()
                            for symbol in symbols:
                                data = data_fetcher.get_stock_data(symbol)
                                if data.empty:
                                    data = fallback.generate_stock_data(symbol, days=252)  # Fallback for testing
                                if not data.empty:
                                    returns_data[symbol] = data['close']
                            
                            # Optimize portfolio
                            optimization_type = {'conservative': 'min_vol', 'moderate': 'sharpe', 'aggressive': 'max_return'}[risk_profile]
                            portfolio_result = optimizer.optimize_portfolio(returns_data, optimization_type=optimization_type)
                            
                            if portfolio_result and portfolio_result.get('success'):
                                logger.info("Portfolio optimization successful")
                                st.success("✅ Portfolio Created Successfully!")
                                
                                # Display allocation
                                st.subheader("📊 Your Portfolio Allocation")
                                allocation_data = []
                                for symbol, weight in portfolio_result['weights'].items():
                                    amount = weight * investment_amount
                                    beta = risk_calc.calculate_beta(symbol, returns_data[symbol])
                                    allocation_data.append({
                                        "Asset": symbol,
                                        "Allocation %": f"{weight*100:.1f}%",
                                        "Amount (₹)": f"₹{amount:,.2f}",
                                        "Type": "Equity",
                                        "Beta": f"{beta:.2f}" if beta is not None else "N/A"
                                    })
                                
                                allocation_df = pd.DataFrame(allocation_data)
                                allocation_df.index = allocation_df.index + 1
                                st.dataframe(allocation_df, use_container_width=True, height=300)
                                
                                # Pie chart
                                st.subheader("📈 Portfolio Visualization")
                                fig = px.pie(
                                    values=[float(row['Allocation %'].replace('%', '')) for row in allocation_data],
                                    names=[row['Asset'] for row in allocation_data],
                                    title=f"{risk_profile.title()} Portfolio - ₹{investment_amount:,.0f}"
                                )
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                fig.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Metrics
                                metrics = portfolio_result['metrics']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Expected Return", f"{metrics['return']:.2%}")
                                with col2:
                                    st.metric("Volatility", f"{metrics['volatility']:.2%}")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                                
                                # Store in DatabaseManager
                                if db.is_available():
                                    db.store_portfolio_recommendation(
                                        user_session=f"portfolio_creation_{datetime.now().isoformat()}",
                                        investment_amount=investment_amount,
                                        risk_profile=risk_profile.upper(),
                                        recommended_allocation=portfolio_result['weights'],
                                        ai_analysis=f"Optimized portfolio for {risk_profile} profile: Return {metrics['return']:.2%}, Volatility {metrics['volatility']:.2%}"
                                    )
                                
                                st.session_state.latest_portfolio = {
                                    'result': portfolio_result,
                                    'amount': investment_amount,
                                    'risk': risk_profile,
                                    'sectors': sectors
                                }
                            else:
                                # This block now handles both failed optimization and None return
                                error_message = portfolio_result.get('message', 'Could not generate a valid portfolio. This may be due to temporary data source issues.') if portfolio_result else "Could not generate a valid portfolio due to data fetching issues."
                                logger.error(f"Portfolio optimization failed: {error_message}")
                                st.error(f"Error creating portfolio: {error_message}")
                                
                        except Exception as e:
                            logger.error(f"Portfolio creation error: {str(e)}")
                            st.error(f"Error creating portfolio: {str(e)}")
                else:
                    logger.warning("User tried to generate portfolio without selecting sectors")
                    st.error("Please select at least one sector.")
        
        elif action == "market_research":
            st.subheader("📊 Indian Market Research")
            
            research_sectors = st.multiselect(
                "Select Sectors for Research:",
                options=['Nifty IT', 'Nifty Bank', 'Nifty FMCG', 'Nifty Auto', 'Nifty Pharma'],
                default=['Nifty IT', 'Nifty Bank']
            )
            
            if st.button("🔍 Conduct Research", type="primary"):
                if research_sectors:
                    logger.info(f"User requesting market research for sectors: {research_sectors}")
                    with st.spinner("Analyzing Indian market sectors..."):
                        try:
                            sector_tickers = {
                                'Nifty IT': '^CNXIT',
                                'Nifty Bank': '^NSEBANK',
                                'Nifty FMCG': '^CNXFMCG',
                                'Nifty Auto': '^CNXAUTO',
                                'Nifty Pharma': '^CNXPHARMA'
                            }
                            research_results = []
                            for sector in research_sectors:
                                ticker = sector_tickers.get(sector)
                                if ticker:
                                    data = data_fetcher.get_stock_data(ticker)
                                    if data.empty:
                                        data = fallback.generate_stock_data(ticker, days=252)
                                    if not data.empty:
                                        analysis = st.session_state.ai_analyzer.get_market_analysis(f"Analyze {sector} sector using data for {ticker}")
                                        research_results.append(f"**{sector}**:\n{analysis}")
                            
                            if research_results:
                                logger.info("Market research completed successfully")
                                st.success("✅ Research Complete")
                                st.markdown("### Market Research Results")
                                for result in research_results:
                                    st.markdown(result)
                                
                                # Store in DatabaseManager
                                if db.is_available():
                                    db.store_portfolio_recommendation(
                                        user_session=f"market_research_{datetime.now().isoformat()}",
                                        investment_amount=0,
                                        risk_profile="RESEARCH",
                                        recommended_allocation={},
                                        ai_analysis="\n".join(research_results)
                                    )
                            else:
                                logger.warning("Market research returned empty results")
                                st.error("Research failed. Please check API connectivity.")
                        except Exception as e:
                            logger.error(f"Market research error: {str(e)}")
                            st.error(f"Research error: {str(e)}")
                else:
                    logger.warning("User tried to conduct research without selecting sectors")
                    st.error("Please select at least one sector.")
        
        elif action == "trading_simulation":
            st.subheader("🎲 AI Trading Simulation")
            
            col_sim1, col_sim2 = st.columns(2)
            
            with col_sim1:
                sim_capital = st.number_input("Simulation Capital (₹)", min_value=100000.0, value=5000000.0, step=100000.0)
                sim_duration = st.selectbox("Simulation Duration", ["1 week", "1 month", "3 months", "6 months"])
            
            with col_sim2:
                sim_risk = st.selectbox("Risk Tolerance", ['conservative', 'moderate', 'aggressive'], index=1)
                sim_assets = st.multiselect("Assets to Trade", 
                                          ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'],
                                          default=['RELIANCE.NS', 'TCS.NS'])
            
            if st.button("🚀 Run Simulation", type="primary"):
                with st.spinner("Running AI trading simulation..."):
                    try:
                        # Initialize TradingEnvironment for simulation
                        from utils.trading_environment import DRLTrainer
                        trainer = DRLTrainer(symbols=sim_assets)
                        training_results = trainer.train_agent(episodes=10, batch_size=32)
                        
                        # Display simulation results
                        st.success("✅ Simulation Complete")
                        st.subheader("Trading Simulation Results")
                        final_result = training_results[-1]
                        st.write(f"Final Portfolio Value: ₹{final_result['final_value']:,.2f}")
                        st.write(f"Total Return: {final_result['total_return']:.2%}")
                        
                        # Plot portfolio value over time
                        returns = [result['final_value'] for result in training_results]
                        fig = px.line(x=range(1, len(returns)+1), y=returns, title="Portfolio Value Over Episodes (₹)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store in DatabaseManager
                        if db.is_available():
                            db.store_portfolio_recommendation(
                                user_session=f"trading_simulation_{datetime.now().isoformat()}",
                                investment_amount=sim_capital,
                                risk_profile=sim_risk.upper(),
                                recommended_allocation={symbol: 1/len(sim_assets) for symbol in sim_assets},
                                ai_analysis=f"Trading simulation: Final value ₹{final_result['final_value']:,.2f}, Return {final_result['total_return']:.2%}"
                            )
                    except Exception as e:
                        logger.error(f"Trading simulation error: {str(e)}")
                        st.error(f"Simulation error: {str(e)}")
        
        elif action == "market_prediction":
            st.subheader("🔮 Indian Market Dynamics Prediction")
            
            prediction_horizon = st.selectbox(
                "Prediction Time Horizon",
                ["1_week", "1_month", "3_months", "6_months", "1_year"],
                index=2
            )
            
            if st.button("🔮 Generate Predictions", type="primary"):
                with st.spinner("Analyzing Indian market patterns..."):
                    try:
                        prompt = f"Predict market dynamics for the Indian stock market (Nifty 50 and key sectors) over {prediction_horizon.replace('_', ' ')}"
                        predictions = st.session_state.ai_analyzer.get_market_analysis(prompt)
                        
                        if predictions and predictions != "AI analysis not available":
                            st.success("✅ Predictions Generated")
                            st.markdown("### Market Dynamics Forecast")
                            st.write(predictions)
                            
                            # Store in DatabaseManager
                            if db.is_available():
                                db.store_portfolio_recommendation(
                                    user_session=f"market_prediction_{datetime.now().isoformat()}",
                                    investment_amount=0,
                                    risk_profile="PREDICTION",
                                    recommended_allocation={},
                                    ai_analysis=predictions
                                )
                        else:
                            st.error("Prediction analysis temporarily unavailable")
                    except Exception as e:
                        logger.error(f"Market prediction error: {str(e)}")
                        st.error(f"Prediction error: {str(e)}")
        
        elif action == "portfolio_analysis":
            st.subheader("📈 Full Portfolio Analysis")
            st.info("Enter your existing investments for comprehensive analysis")
            
            portfolio_name = st.text_input("Portfolio Name", value="My Indian Portfolio")
            
            if 'manual_portfolio' not in st.session_state:
                st.session_state.manual_portfolio = {}
            
            st.markdown("#### Add Your Investments")
            st.info("Enter the ticker (e.g., RELIANCE.NS) and amount invested in INR")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                investment_name = st.text_input(
                    "Investment Ticker",
                    help="e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS"
                )
            
            with col2:
                investment_amount = st.number_input(
                    "Amount Invested (₹)",
                    min_value=0.0,
                    step=10000.0
                )
            
            with col3:
                if st.button("➕ Add Investment", key="add_investment"):
                    if investment_name and investment_amount > 0:
                        st.session_state.manual_portfolio[investment_name] = investment_amount
                        st.success(f"Added {investment_name}: ₹{investment_amount:,.2f}")
                        st.rerun()
                    else:
                        st.error("Please enter both ticker and amount")
            
            if st.session_state.manual_portfolio:
                st.markdown("#### Current Portfolio Holdings")
                total_value = 0
                for name, amount in st.session_state.manual_portfolio.items():
                    col_name, col_amount, col_remove = st.columns([3, 1, 1])
                    with col_name:
                        st.write(f"**{name}**")
                    with col_amount:
                        st.write(f"₹{amount:,.2f}")
                    with col_remove:
                        if st.button("🗑️", key=f"remove_{name}"):
                            del st.session_state.manual_portfolio[name]
                            st.rerun()
                    total_value += amount
                
                st.markdown(f"### **Total Portfolio Value: ₹{total_value:,.2f}**")
            
            if st.session_state.manual_portfolio and st.button("🔍 Analyze My Portfolio", type="primary"):
                with st.spinner("Analyzing your Indian portfolio..."):
                    try:
                        holdings = st.session_state.manual_portfolio
                        holdings_summary = [f"- {name}: ₹{amount:,.2f}" for name, amount in holdings.items()]
                        
                        analysis_prompt = f"""
                        Analyze this ₹{total_value:,.2f} Indian portfolio:

                        HOLDINGS:
                        {chr(10).join(holdings_summary)}

                        Provide analysis in this structure:

                        MARKET OUTLOOK:
                        Brief assessment of Indian market conditions (Nifty 50, key sectors).

                        INDIVIDUAL ASSETS:
                        For each holding, provide: Current assessment, outlook, and recommendation.

                        FUTURE VALUE PREDICTIONS:
                        1-Year Target: ₹ amount (assumptions)
                        5-Year Target: ₹ amount (assumptions)
                        10-Year Target: ₹ amount (assumptions)
                        Include best/likely/worst case for each timeframe.

                        INVESTMENT RECOMMENDATIONS:
                        STOCKS TO BUY:
                        1. [Ticker] - Company Name - ₹ allocation - Reason
                        2. [Ticker] - Company Name - ₹ allocation - Reason
                        3. [Ticker] - Company Name - ₹ allocation - Reason

                        REBALANCING PLAN:
                        - What to sell: specific amounts
                        - What to buy: specific amounts
                        - Target percentages for each asset

                        Keep responses concise and actionable. Use INR for all amounts.
                        """
                        
                        analysis = st.session_state.ai_analyzer.get_market_analysis(analysis_prompt)
                        
                        if analysis and analysis != "AI analysis not available":
                            st.success("✅ Comprehensive Analysis Complete")
                            
                            # Portfolio summary
                            st.markdown("#### Portfolio Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Value", f"₹{total_value:,.2f}")
                            with col2:
                                st.metric("Number of Holdings", len(holdings))
                            
                            # Holdings breakdown
                            st.markdown("#### Holdings Breakdown")
                            for name, amount in holdings.items():
                                percentage = (amount / total_value) * 100
                                st.write(f"• **{name}**: ₹{amount:,.2f} ({percentage:.1f}%)")
                            
                            # Full analysis
                            st.markdown("### Portfolio Analysis")
                            with st.expander("📊 Complete Analysis", expanded=True):
                                st.text_area("Raw Analysis", analysis, height=400, disabled=True)
                            
                            # Parsed sections
                            parsed_sections = parse_analysis_sections(analysis)
                            for section, content in parsed_sections.items():
                                st.markdown(f"**{section.replace('_', ' ').title()}**")
                                st.write(content)
                                st.markdown("---")
                            
                            # Store in DatabaseManager
                            if db.is_available():
                                db.store_portfolio_recommendation(
                                    user_session=f"portfolio_analysis_{datetime.now().isoformat()}",
                                    investment_amount=total_value,
                                    risk_profile="ANALYSIS",
                                    recommended_allocation={name: amount/total_value for name, amount in holdings.items()},
                                    ai_analysis=analysis
                                )
                            
                            st.session_state.analyzed_manual_portfolio = {
                                'name': portfolio_name,
                                'holdings': holdings,
                                'total_value': total_value,
                                'analysis': analysis,
                                'analysis_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                            }
                        else:
                            st.error("Analysis temporarily unavailable. Please check API connectivity.")
                    except Exception as e:
                        logger.error(f"Portfolio analysis error: {str(e)}")
                        st.error(f"Analysis error: {str(e)}")
            
            if 'analyzed_manual_portfolio' in st.session_state:
                prev_analysis = st.session_state.analyzed_manual_portfolio
                st.markdown("#### Previous Analysis")
                st.info(f"Analysis for '{prev_analysis['name']}' - {prev_analysis.get('analysis_date', 'Unknown date')}")
                if st.button("📋 Show Previous Analysis"):
                    st.markdown("### Previous Portfolio Analysis")
                    st.write(prev_analysis['analysis'])
    
    # Database status in sidebar
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

def parse_analysis_sections(analysis):
    """Parse analysis text into structured sections"""
    sections = {}
    
    section_markers = {
        "MARKET OUTLOOK:": "market_outlook",
        "INDIVIDUAL ASSETS:": "individual_assets",
        "FUTURE VALUE PREDICTIONS:": "future_predictions",
        "INVESTMENT RECOMMENDATIONS:": "investment_recommendations",
        "REBALANCING PLAN:": "rebalancing_plan"
    }
    
    section_positions = []
    for marker, key in section_markers.items():
        pos = analysis.upper().find(marker.upper())
        if pos != -1:
            section_positions.append((pos, marker, key))
    
    section_positions.sort()
    
    for i, (pos, marker, key) in enumerate(section_positions):
        start = pos + len(marker)
        end = section_positions[i+1][0] if i < len(section_positions)-1 else len(analysis)
        content = analysis[start:end].strip()
        if content:
            sections[key] = content
    
    return sections

if __name__ == "__main__":
    main()