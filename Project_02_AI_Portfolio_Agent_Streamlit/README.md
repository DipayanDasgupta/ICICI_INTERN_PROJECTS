# Project 2: AI Portfolio Agent (Streamlit)

This project is a real-time, AI-driven portfolio management assistant built with **Streamlit**. It serves as an intelligent dashboard and interactive agent for investors focused on the Indian stock market. It provides live market data, on-demand portfolio optimization, trading simulations using Deep Reinforcement Learning, and comprehensive portfolio analysis powered by Google Gemini.

## 🚀 Key Features

-   **Multi-Page Streamlit Interface**: A clean, interactive UI with dedicated sections for:
    -   **Indian Market Analysis**: A live dashboard tracking major indices (Nifty 50, Sensex) and key sectors (IT, Banking, FMCG) with real-time price data and AI-generated market summaries.
    -   **AI Portfolio Agent**: An interactive agent that can perform several actions:
        -   **Create Optimal Portfolio**: Generates a diversified portfolio based on user-defined capital and risk profile.
        -   **Conduct Market Research**: Provides AI-driven analysis of selected market sectors.
        -   **Run Trading Simulation**: A framework for testing AI-based trading strategies using an OpenAI Gym environment and a Deep Reinforcement Learning (DRL) agent built with TensorFlow.
        -   **Predict Market Dynamics**: Generates long-term market forecasts.
        -   **Full Portfolio Analysis**: Analyzes a user's manually entered portfolio, providing an AI-driven report on its composition, risks, and recommended actions.
    -   **Data Management**: An interface to view and query the data stored in the local SQLite database.
-   **Real-Time Data & Fallbacks**: Primarily uses `yfinance` to fetch live market data. If an API fails, it gracefully falls back to generating realistic synthetic data for demonstration purposes, ensuring the application remains functional.
-   **Quantitative Engine**: Integrates logic for portfolio optimization (Sharpe Ratio, Min Variance), risk calculation (Beta, VaR, CVaR), and a quantitative stock screener.
-   **Database Caching**: Uses a local SQLite database to cache market data and store portfolio recommendations, improving performance and persisting results.

## ⚙️ Setup and Usage

### Prerequisites
- Python >= 3.10
- Git

### Installation
1.  **Navigate to this project's directory:**
    ```bash
    cd Project_02_AI_Portfolio_Agent_Streamlit
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    - Create a `.env` file in this directory (`Project_02_AI_Portfolio_Agent_Streamlit`).
    - Add your API keys to it:
      ```env
      GEMINI_API_KEY="your-google-gemini-api-key"
      FINNHUB_API_KEY="your-finnhub-api-key"
      # DATABASE_URL is optional, defaults to a local SQLite file
      ```

### Running the Application
1.  **Launch the Streamlit server:**
    ```bash
    streamlit run app.py
    ```

2.  The application will open automatically in your web browser, typically at **http://localhost:8501**.
