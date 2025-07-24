# Project 1: AI-Powered Portfolio Optimizer & Backtesting Engine

This is an end-to-end, professional-grade web application for **quantitative investment strategy development**, **robust backtesting**, and **advanced portfolio analysis**. This platform enables a full-cycle experience — from strategy creation to performance interpretation — through an intuitive UI, a walk-forward backtesting engine, and AI-driven portfolio diagnostics.

## 🚀 Key Features

-   **Asynchronous Architecture**: Utilizes **Celery** and **Redis** to run all heavy computations (ML predictions, backtests) in the background, ensuring a smooth, non-blocking user experience.
-   **Advanced Backtesting Engine**:
    -   **ML-Driven Strategy**: Periodically retrains a LightGBM model on a rolling window of data to select stocks based on predictive quantitative features.
    -   **Custom Portfolio Rebalancing**: Backtest any user-defined portfolio with dynamic rebalancing using **Hierarchical Risk Parity (HRP)**.
    -   **Walk-Forward Methodology**: Eliminates lookahead bias by only using historical data available at each point in time, simulating real-world strategy deployment.
    -   **Indian Market Factor Analysis**: Decomposes portfolio returns against the Fama-French four factors (Market, Size, Value, Momentum) using data tailored for the Indian market.
-   **Comprehensive Reporting & AI Insights**:
    -   Generates interactive reports with 30+ KPIs (Sharpe, Sortino, Calmar, etc.), equity curves, and drawdown charts using **QuantStats** and **Plotly.js**.
    -   **Google Gemini Integration**: Provides AI-powered natural-language summaries of backtest performance, explains complex financial metrics, and analyzes portfolio risk concentrations through hierarchical clustering.
-   **Portfolio & Universe Studio**:
    -   Interactively build, save, and manage custom stock portfolios.
    -   Upload and manage custom stock universes from CSV files, allowing backtests on specific watchlists or thematic baskets.
-   **Local Data First**: All operations run on a local **SQLite** database, making the application fast, self-contained, and suitable for offline analysis. Standalone scripts are provided to build this database from scratch using `yfinance`.

## ⚙️ Setup and Usage

### Prerequisites
- Python >= 3.10
- Git
- Redis Server (must be running for the application to work)

### Installation
1.  **Navigate to this project's directory:**
    ```bash
    cd Project_01_AI_Portfolio_Optimizer_Flask
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

4.  **Configure Google Gemini (Optional but Recommended):**
    - For AI-powered reporting features, you need a Google Gemini API key.
    - Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Set it as an environment variable:
      ```bash
      export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      ```

### Build Local Database & Train ML Model
These scripts are located in the `standalone_utilities` directory. Run them from the `Project_01_AI_Portfolio_Optimizer_Flask` root.

1.  **Ingest Data (Run this first, can take ~20-30 mins):**
    This script downloads 10 years of stock data and Indian market factor data into local SQLite databases.
    ```bash
    python standalone_utilities/data_ingestion.py
    ```

2.  **Train the Production ML Model:**
    This script uses the newly created database to train the LightGBM model and saves it for the web app to use.
    ```bash
    python standalone_utilities/train_and_save_model.py
    ```

### Running the Application
You need **three separate terminals** running in the `Project_01_AI_Portfolio_Optimizer_Flask` directory. Ensure your virtual environment is activated in each.

1.  **Terminal 1: Start Redis Server**
    ```bash
    redis-server
    ```

2.  **Terminal 2: Start Celery Worker**
    The worker waits for and executes background tasks.
    ```bash
    celery -A run.celery worker -l info --pool=solo
    ```
    *Note: `--pool=solo` is recommended for Windows compatibility.*

3.  **Terminal 3: Start Flask Web Server**
    ```bash
    flask run --port=5001
    ```

Now, open your web browser and navigate to **http://127.0.0.1:5001**.
