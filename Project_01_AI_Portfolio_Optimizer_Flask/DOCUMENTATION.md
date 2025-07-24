

***

# Developer Documentation

This document provides a detailed, file-by-file, and function-by-function breakdown of the entire AI-Powered Quantitative Investment Platform. It is intended for developers who want to understand, maintain, or extend the platform's functionality.

## Table of Contents

1.  [**High-Level Architecture**](#1-high-level-architecture)
    *   [Core Philosophy](#core-philosophy)
    *   [Request Flow: Asynchronous Operations](#request-flow-asynchronous-operations)
2.  [**Project Setup & Standalone Scripts**](#2-project-setup--standalone-scripts)
    *   [`run.py`](#runpy)
    *   [`data_ingestion.py`](#data_ingestionpy)
    *   [`add_factors.py`](#add_factorspy)
    *   [`train_and_save_model.py`](#train_and_save_modelpy)
    *   [`standalone_backtest.py`](#standalone_backtestpy)
    *   [`tune_model.py`](#tune_modelpy)
3.  [**Core Application (`app/`)**](#3-core-application-app)
    *   [**Application Setup & Configuration**](#application-setup--configuration)
        *   [`app/__init__.py`](#app__init__py)
        *   [`app/config.py`](#appconfigpy)
    *   [**Backend Logic: Routes & Tasks**](#backend-logic-routes--tasks)
        *   [`app/routes.py`](#approutespy)
        *   [`app/tasks.py`](#apptaskspy)
    *   [**Data & Machine Learning Core**](#data--machine-learning-core)
        *   [`app/data_fetcher.py`](#appdata_fetcherpy)
        *   [`app/strategy.py`](#appstrategypy)
        *   [`app/ml_models.py`](#appml_models_py)
        *   [`app/backtesting.py`](#appbacktestingpy)
        *   [`app/factor_analysis.py`](#appfactor_analysispy)
    *   [**AI & Reporting**](#ai--reporting-1)
        *   [`app/reporting.py`](#appreportingpy)
4.  [**Frontend (`app/static/` & `app/templates/`)**](#4-frontend-appstatic--apptemplates)
    *   [`app/templates/`](#apptemplates-1)
    *   [`app/static/js/main.js`](#appstaticjsmainjs)
5.  [**Database Schema**](#5-database-schema)
    *   [`market_data.db`](#market_datadb)
    *   [`user_portfolios.db`](#user_portfoliosdb)

---

## 1. High-Level Architecture

### Core Philosophy

The application is built around a **client-server model** with a crucial enhancement: **asynchronous task processing**. The frontend (client) is a single-page application (SPA) experience built with HTML, Bootstrap, and JavaScript. The backend is a Python Flask server.

The key architectural decision is to **never block the user interface**. Any computation that takes more than a second (like ML model runs or backtests) is offloaded to a background worker process managed by **Celery** and **Redis**.

### Request Flow: Asynchronous Operations

1.  **User Action**: The user clicks a button like "Run Backtest" in the browser.
2.  **API Request**: The frontend JavaScript sends a request with the configuration data to a Flask API endpoint (e.g., `/api/run_backtest`).
3.  **Task Dispatch**: The Flask route **does not** perform the calculation. Instead, it immediately dispatches a job (a "task") to the Celery worker via the Redis message broker and instantly returns a unique `task_id` to the browser.
4.  **Polling for Status**: The frontend JavaScript receives the `task_id` and starts "polling" a status endpoint (e.g., `/api/backtest_status/<task_id>`) every few seconds.
5.  **Background Processing**: The Celery worker, running in a separate process, picks up the job from Redis. It executes the long-running Python function (e.g., `run_backtest`). As it works, it can update its status (e.g., "Loading data...", "Retraining model...") back to Redis.
6.  **Status Updates**: When the frontend polls the status endpoint, Flask retrieves the latest status from Redis and sends it back. The UI can then display a live progress bar or status message.
7.  **Final Result**: Once the Celery worker finishes the task, it stores the final result (typically a large JSON object) in Redis.
8.  **Result Retrieval**: On the next poll, the Flask status endpoint sees that the task is `SUCCESS`, retrieves the final result from Redis, and sends it to the frontend.
9.  **UI Update**: The frontend JavaScript receives the complete data payload and dynamically updates the page—rendering charts, populating tables, and displaying KPIs.

---

## 2. Project Setup & Standalone Scripts

These are top-level Python scripts executed from the command line to initialize, run, or debug the application.

### `run.py`

**Purpose**: The main entry point for running the Flask application and for the Celery worker to identify the app instance.

-   **`app = create_app()`**: This line invokes the application factory from `app/__init__.py` to construct and configure the Flask app instance.
-   **`celery = app.extensions["celery"]`**: After the app is created, this crucial line extracts the Celery instance that was initialized and attached to the app object. The Celery CLI command (`celery -A run.celery worker`) specifically looks for this `celery` object within `run.py` to start the worker.
-   **`if __name__ == '__main__':`**: This block is only executed when you run `flask run` or `python run.py`. It starts the Flask development server. It is *not* executed when the Celery worker starts.

### `data_ingestion.py`

**Purpose**: A one-time utility script to build and populate the local SQLite databases from scratch.

-   **`create_database()`**: Ensures that both the market data database (`market_data.db`) and the user portfolios database (`user_portfolios.db`) exist. It creates all necessary tables with their schemas if they are missing. This makes the script idempotent (safe to run multiple times).
-   **`ingest_data()`**: The main data fetching function. It fetches up to 10 years of historical stock data for all symbols defined in the `NIFTY_500` universe from the `yfinance` library.
    -   **Robustness**: It intelligently checks for symbols already in the database to avoid re-downloading data, allowing the script to resume if interrupted. It includes a retry mechanism for failed downloads and gracefully handles symbols with no available data.
    -   **Data Cleaning**: It correctly handles the `Adj Close` column to ensure split-adjusted prices are used for `Close` and fetches metadata like the stock's `Sector`.
-   **`if __name__ == '__main__':`**: The main execution block. It first calls `create_database()`, then `ingest_data()` to populate stock prices, and finally `ingest_fama_french_data()` to populate the factor data, resulting in a complete, ready-to-use set of databases.

### `add_factors.py`

**Purpose**: A simple, standalone utility to populate the `fama_french_factors` table in the database from the local `2025-03_FourFactors...csv` file. This is separated from the main data ingestion to allow for independent updates if the factor data file changes.

-   **`main()`**: The script's entry point, which simply calls the `ingest_fama_french_data` function from the `app.factor_analysis` module to perform the data ingestion.

### `train_and_save_model.py`

**Purpose**: A one-time script to train the production machine learning model on the entire local dataset and save the trained model object for the web app to use.

-   **`train_production_model(symbols)`**: Orchestrates the training process. It fetches data for a given list of symbols from the local `market_data.db`, generates all features using the `generate_all_features` function, and trains a `LightGBM` regressor on the complete, combined dataset.
-   **`run_training_pipeline()`**: The main workflow. It defines the universe to train on (`NIFTY_500`), filters out any symbols that were found to have no data during ingestion, calls `train_production_model`, and then serializes and saves the resulting model object to `app/stock_selector_model.joblib` using `joblib`.

### `standalone_backtest.py`

**Purpose**: A developer utility for running a backtest entirely from the command line, bypassing the web interface and local database. This is excellent for quickly debugging the core strategy logic or testing on live data.

-   **`run_backtest()`**: A self-contained function that:
    1.  Fetches live data directly from `yfinance`.
    2.  Runs the full walk-forward backtesting loop (model retraining, prediction, optimization).
    3.  Calculates portfolio returns.
    4.  Generates a static HTML report using the `QuantStats` library.

### `tune_model.py`

**Purpose**: A developer script to perform hyperparameter tuning for the `LightGBM` model using the `Optuna` library.

-   **`objective(trial, X, y)`**: The core function that `Optuna` calls on each trial. It defines the search space for each hyperparameter (e.g., `lambda_l1`, `num_leaves`), trains a model with the trial parameters, evaluates its performance (R^2 score), and returns the score for Optuna to maximize.
-   **`run_tuning()`**: The main function that prepares the dataset, creates an `Optuna` study, runs the optimization process for a set number of trials, and prints the best-found parameters at the end. The output of this script was used to set the `best_params` in `train_and_save_model.py`.

---

## 3. Core Application (`app/`)

This directory contains all the source code for the Flask application itself.

### **Application Setup & Configuration**

#### `app/__init__.py`

**Purpose**: Contains the **Application Factory**. This pattern is essential for creating well-structured, testable, and scalable Flask applications.

-   **`celery_init_app(app: Flask) -> Celery`**: A helper function that takes a Flask app instance and configures a Celery instance to work with it. It sets the broker and result backend from the Flask config and ensures that Celery tasks run within the Flask application context, giving them access to `current_app` and its extensions.
-   **`create_app()`**: The factory function. It performs the following steps in order:
    1.  Creates the `Flask` app instance.
    2.  Sets up the configuration for `Celery`.
    3.  Sets the path for the portfolios database file in the app's config.
    4.  Checks if the ML model file (`stock_selector_model.joblib`) exists to determine if live analysis can be enabled.
    5.  Initializes `Celery` using `celery_init_app`.
    6.  **Crucially**, it imports the `routes` and `tasks` modules *after* the app is created to prevent circular import errors.
    7.  Returns the fully configured `app` object.

#### `app/config.py`

**Purpose**: A central, static configuration hub.

-   **`DB_FILE`, `PORTFOLIOS_DB_FILE`**: Defines the string file paths for the two SQLite databases. Using a config file makes it easy to change these paths for testing or deployment.
-   **`STOCK_UNIVERSES`**: A Python dictionary containing the static, hardcoded lists of stock symbols for various NSE indices (e.g., `NIFTY_50`, `NIFTY_500`, etc.). This is the primary source for the application's pre-defined universes.

### **Backend Logic: Routes & Tasks**

#### `app/routes.py`

**Purpose**: Defines all the web routes (pages) and API endpoints that the frontend JavaScript communicates with.

-   **`@app.route('/')` (index)**: Renders the main `index.html` page. It fetches the complete list of all available universes (both static from `config.py` and custom from the database) to populate the dropdown menus in the UI.
-   **`/api/analyze_and_optimize` (POST)**: **Starts the live analysis background task**. It receives the user's configuration from the UI, validates that the ML model exists, and dispatches a `run_live_analysis_task` to the Celery worker. It immediately returns a `task_id` to the frontend.
-   **`/api/analysis_status/<task_id>` (GET)**: **Polls for live analysis results**. The frontend calls this endpoint repeatedly with the `task_id`. The route checks the task's state in Redis and returns its current status (`PENDING`, `PROGRESS`, `SUCCESS`, `FAILURE`) and, upon completion, the final results payload.
-   **`/api/portfolios` (GET, POST)** & **`/api/portfolios/<id>` (PUT, DELETE)**: A set of standard RESTful API endpoints for performing CRUD (Create, Read, Update, Delete) operations on user-saved portfolios stored in the `user_portfolios.db`.
-   **`/api/universes` (GET, POST)** & **`/api/universes/<id>` (PUT, DELETE)**: Similar CRUD endpoints for managing user-uploaded custom stock universes. The POST method specifically handles the parsing and saving of uploaded CSV files.
-   **`/api/run_backtest` (POST)**: **Starts a backtesting background task**. Based on the user's config, it determines whether to run the ML-driven strategy or a custom portfolio backtest and dispatches the appropriate task (`run_backtest_task` or `run_custom_backtest_task`) to Celery, returning a `task_id`.
-   **`/api/backtest_status/<task_id>` (GET)**: **Polls for backtest results**. This provides real-time progress updates on the backtest and returns the final, large JSON report upon completion.
-   **`/api/universe_stocks/<universe_name>` (GET)**: A helper endpoint used by the Portfolio Studio. It returns a sorted list of stock symbols for any given universe name, correctly handling both static and custom universes by calling the `get_stock_universe` function.
-   **`/api/analyze_clusters` (POST)**: The endpoint for portfolio cluster analysis. It receives a list of stocks from the UI, calls the backend functions to perform the analysis, generates the AI redundancy report, and returns the final data needed to render the dendrogram chart and the AI text.
-   **`/api/ask_chatbot` (POST)**: Powers the AI chatbot. It receives the user's question and the current backtest context from the UI, passes them to the `answer_user_question` reporting function, and returns the AI's response.
-   **`/api/generate_pdf` (POST)**: Receives the HTML content of the backtest report from the client, uses the `WeasyPrint` library to convert it into a PDF file in memory, and returns it to the user for download.
-   **`/api/explain_factors` (POST)**: A simple endpoint that calls the AI reporting function to generate a definition of the Fama-French factors and returns it to be displayed in a modal.

#### `app/tasks.py`

**Purpose**: Defines the long-running functions that are executed asynchronously by the Celery worker. **This is where all the heavy computation happens.**

-   **`@shared_task(bind=True)`**: This decorator registers the function as a Celery task. `bind=True` gives the function access to `self`, which allows it to update its own state (e.g., `self.update_state(...)`).
-   **`run_live_analysis_task(...)`**: The background job for the "Live Analysis" tab. It loads the ML model, calls `predict_top_stocks`, performs the chosen portfolio optimization, generates the rationale and other data, and carefully prepares a **JSON-serializable** dictionary to be sent back as the final result. It uses `self.update_state` to provide live status messages to the UI.
-   **`run_backtest_task(...)`**: The background job for the ML-driven backtest. It's a wrapper around the core `run_backtest` function from the `backtesting` module. Its main role is to pass the `progress_callback` function, which allows `run_backtest` to send live progress updates back to the UI via the task's state.
-   **`run_custom_backtest_task(...)`**: The background job for the custom portfolio backtest. It first fetches the portfolio's holdings from the database and then calls the `run_custom_portfolio_backtest` function, also passing along the progress callback for UI updates.

### **Data & Machine Learning Core**

#### `app/data_fetcher.py`

**Purpose**: The dedicated data access layer (DAL). This is the *only* module that should directly interact with the SQLite databases.

-   **`get_stock_universe(universe_name)`**: A versatile function that can fetch a list of stock symbols from two different sources. It first checks if the `universe_name` is a key in the static `STOCK_UNIVERSES` dictionary (e.g., "NIFTY_50"). If not, it assumes the name is for a custom universe (e.g., "custom_1") and queries the `custom_universes` table in the database to retrieve the list of symbols.
-   **`get_historical_data(symbol, start_date, end_date)`**: Queries the `market_data.db` for raw historical price data for a single symbol within a given date range. After fetching the price data, it performs a second query to the `stock_metadata` table to get the stock's sector and enriches the DataFrame with this information before returning it. It also handles data type conversion to ensure prices are numeric and dates are datetime objects.

#### `app/strategy.py`

**Purpose**: Contains the feature engineering logic for the machine learning model.

-   **`generate_all_features(df: pd.DataFrame, benchmark_df: pd.DataFrame)`**: This function is the heart of the predictive strategy. It takes a stock's raw price DataFrame and the benchmark's price DataFrame and calculates a wide range of technical and quantitative features. This includes:
    -   Simple features like Moving Averages (`MA_20`, `MA_50`) and Rate-of-Change (`ROC_20`).
    -   Volatility and RSI.
    -   More advanced features like multi-period Momentum and a rolling 3-month Sharpe Ratio.
    -   Crucially, it calculates `Relative_Strength` against the benchmark in a way that avoids lookahead bias.
    -   Finally, it computes the **`Target`** variable, which is the 22-day (approx. 1 month) forward return. This is the value the ML model is trained to predict.

#### `app/ml_models.py`

**Purpose**: Contains the core machine learning prediction logic and the various portfolio optimization algorithms.

-   **`predict_top_stocks(model, symbols, top_n)`**: Uses the pre-trained and loaded ML model to predict the `Target` (forward return) for a list of symbols. It then ranks them and returns the `top_n` stocks with the highest predicted scores. It also returns the full ranked list and the model's feature importances for display in the UI.
-   **`optimize_portfolio(portfolio_data, risk_free_rate)`**: Implements the classic **Markowitz Mean-Variance Optimization (MVO)** from Modern Portfolio Theory (MPT). It uses `scipy.optimize.minimize` to find the portfolio weights that maximize the Sharpe Ratio, subject to constraints (weights must sum to 1, and no single stock can have more than 10% allocation).
-   **`optimize_hrp_portfolio(portfolio_data, ...)`**: Implements the standard **Hierarchical Risk Parity (HRP)** algorithm using the `pypfopt` library. After getting the initial HRP weights, it applies a post-processing step to enforce the 10% maximum weight constraint by capping overweight stocks and proportionally redistributing the excess weight to the others.
-   **`optimize_hrp_with_sector_constraints(...)`**: A more advanced, two-stage HRP implementation designed to handle user-defined sector bounds.
    1.  **Stage 1 (Sector Level)**: It first performs an optimization to determine the optimal allocation of capital *between sectors*, minimizing the overall portfolio variance at the sector level while respecting the user's min/max constraints.
    2.  **Stage 2 (Stock Level)**: It then runs the standard HRP algorithm *within* each sector to determine how to allocate that sector's capital among its constituent stocks.
-   **`analyze_portfolio_clusters(portfolio_data)`**: Performs hierarchical clustering on a portfolio's daily returns to identify groups of stocks that move together. It uses these clusters to generate a Plotly dendrogram figure for visualization and programmatically identifies both high-level "Main Clusters" and highly-correlated "Tight Clusters" (for redundancy analysis).
-   **Helper Functions**: Includes `get_portfolio_data` to fetch historical data for a list of symbols and `get_portfolio_sector_exposure` to calculate the percentage weight of each sector in a given portfolio.

#### `app/backtesting.py`

**Purpose**: This module contains the two main backtesting engines. These are complex, stateful functions that simulate strategy performance over time.

-   **`run_backtest(...)`**: The engine for the **ML-driven strategy**. It performs a full walk-forward backtest. It iterates through time, month by month (or at a given rebalance frequency). In each iteration, it:
    1.  **Retrains** the `LightGBM` model on a rolling window of the most recent past data (e.g., 3 years). This ensures the model adapts to changing market conditions.
    2.  Applies a **market regime filter** (checking if the NIFTY 50 is above its 200-day moving average) to decide whether to invest or hold cash for the period.
    3.  Uses the newly trained model to **predict** forward returns and select the top N stocks.
    4.  **Optimizes** the selected portfolio using the chosen method (HRP with constraints or MPT).
    5.  Records the resulting portfolio holdings and logs the action taken.
-   **`run_custom_portfolio_backtest(...)`**: The engine for backtesting **user-defined portfolios**. It is simpler than the ML engine. For each rebalance date, it takes the user's pre-defined list of stocks, filters them against the selected backtest universe, and then dynamically rebalances their weights using the HRP algorithm to manage risk.
-   **`calculate_performance(...)` & `generate_report_payload(...)`**: These are crucial reporting functions called at the end of any backtest. They take the final timeseries of historical holdings, calculate the portfolio's daily returns (including a penalty for transaction costs), and then use the `QuantStats` library and other helpers to compute over 30 different KPIs. They assemble the final, massive JSON object containing all charts, tables, KPIs, logs, and AI reports that gets sent back to the frontend for display.

#### `app/factor_analysis.py`

**Purpose**: Handles all statistical analysis related to the Fama-French-Carhart four-factor model, tailored for the Indian market.

-   **`ingest_fama_french_data()`**: A utility function (used by `data_ingestion.py` and `add_factors.py`) to read the local Indian factor data CSV, clean the data, rename columns for consistency (e.g., WML to UMD), and store it in the `fama_french_factors` table in the main database.
-   **`analyze_factor_exposure(portfolio_returns)`**: This function performs a multiple linear regression. It regresses the portfolio's excess returns (portfolio return - risk-free rate) against the four factors (Market, Size, Value, Momentum). It returns the key results of this regression:
    -   **Alpha**: The portion of the return not explained by the factors (a measure of skill).
    -   **Betas**: The sensitivity (exposure) of the portfolio to each of the four factors.
    -   **R-Squared**: The percentage of the portfolio's movement that is explained by the factors.
-   **`analyze_rolling_factor_exposure(portfolio_returns, ...)`**: This performs the same regression but on a rolling window of data (defaulting to 1 year/252 days). This is computationally intensive but reveals how the portfolio's factor exposures (its fundamental "style") changed over the course of the backtest. The results are used to generate the factor distribution boxplots.

### **AI & Reporting**

#### `app/reporting.py`

**Purpose**: This module is the sole interface to the **Google Gemini Pro API**. It contains several specialized functions, each with a carefully crafted prompt to coax the desired output from the language model.

-   **`generate_gemini_report(...)`**: Takes the key performance indicators and logs from a backtest, formats them into a detailed, structured prompt, and asks the Gemini model to act as a quantitative analyst and provide a qualitative evaluation of the strategy's performance, risks, and potential areas for improvement.
-   **`generate_factor_explanation()`**: Asks the model to act as a finance professor and provide a clear, educational explanation of the four Fama-French factors for a non-expert audience.
-   **`answer_user_question(question, context_data)`**: Powers the interactive chatbot. It takes the user's question and a JSON object of the entire current backtest report as context. The prompt strictly instructs the model to answer the question based *only* on the provided context, preventing hallucination.
-   **`generate_cluster_elimination_report(...)`**: Takes the results from the cluster analysis, including the different cluster groups and the quantitative features of the stocks within them. It then asks the model to act as a risk analyst, interpret the portfolio's risk structure, and recommend specific stocks for elimination from redundant groups based on their weaker metrics.

---

## 4. Frontend (`app/static/` & `app/templates/`)

### `app/templates/`

**Purpose**: These files define the HTML structure and layout of the web application using the Jinja2 templating engine.

-   **`base.html`**: The main template skeleton. It contains the `<!DOCTYPE html>`, `head`, and `body` tags. It imports the Bootstrap CSS and JS, includes the main `main.js` file, defines the top navigation bar, and provides a `{% block content %}{% endblock %}` placeholder where the content from other templates will be injected.
-   **`index.html`**: The primary content file for the entire application. It extends `base.html` and contains all the HTML markup for the tabbed interface (Live Analysis, Portfolio Studio, Universe Studio, Backtesting) and all the Bootstrap modals used for pop-ups (Metrics Help, Cluster Analysis, etc.). It uses `id` attributes extensively on elements (like `divs`, `buttons`, and `tables`) that need to be targeted and manipulated by the JavaScript code. The Jinja templating is used here to dynamically populate the universe dropdowns when the page is first loaded.

### `app/static/js/main.js`

**Purpose**: This is the single, large JavaScript file that controls all frontend interactivity, communication with the Flask API, and dynamic rendering of charts and data.

-   **`document.addEventListener('DOMContentLoaded', ...)`**: This is the main entry point for the script. The entire code is wrapped inside this event listener, which ensures that the script only runs after the full HTML page has been loaded and parsed by the browser. It is responsible for attaching all other event listeners to buttons, forms, and inputs.
-   **Asynchronous API Calls (`fetch`)**: The file makes extensive use of the `fetch` API with `async/await` syntax to communicate with the backend. This is used for starting tasks, polling for status, and retrieving data.
-   **Dynamic UI Updates**: The core responsibility of this file is to manipulate the DOM (Document Object Model). It dynamically creates HTML for tables (`createReturnsTable`), updates the text of KPI cards, and clears/shows/hides different sections of the UI based on the application's state.
-   **Plotly Integration**: It directly interfaces with the `Plotly.js` library. Functions like `plotPortfolioPie()` and the `displayBacktestResults()` function take the JSON data received from the backend and use `Plotly.newPlot()` to render the interactive charts.
-   **Event Handling**: It contains dozens of event listeners (`.addEventListener`) for various user actions:
    -   Clicking buttons (`runAnalysisBtn`, `savePortfolioBtn`, `runBacktestBtn`).
    -   Submitting forms (`chatForm`, `universeUploadForm`).
    -   Changing dropdown selections (`backtestTypeSelector`, `studioUniverseSelector`).
    -   Opening modals (`clusterAnalysisModal`).
-   **State Management**: It maintains some client-side state in variables like `currentBacktestResults` and `lastBacktestLogs` to support features like downloading CSVs or generating PDFs of the most recent report.

---

## 5. Database Schema

### `market_data.db`

-   **`historical_prices`**: The main table for time-series data.
    -   `Date` (TEXT, PK)
    -   `Symbol` (TEXT, PK)
    -   `Open`, `High`, `Low`, `Close` (REAL)
    -   `Volume` (INTEGER)
-   **`stock_metadata`**: Stores static information about each stock.
    -   `Symbol` (TEXT, PK)
    -   `Sector` (TEXT)
-   **`fama_french_factors`**: Stores the daily Indian market factor data.
    -   `Date` (TEXT, PK)
    -   `Mkt-RF`, `SMB`, `HML`, `UMD`, `RF` (REAL)

### `user_portfolios.db`

-   **`custom_portfolios`**: Stores user-saved portfolios.
    -   `id` (INTEGER, PK, AUTOINCREMENT)
    -   `name` (TEXT, UNIQUE)
    -   `stocks_json` (TEXT): A JSON string representing a dictionary of `{ "Symbol": weight }`.
    -   `created_at` (TIMESTAMP)
-   **`custom_universes`**: Stores user-uploaded stock universes.
    -   `id` (INTEGER, PK, AUTOINCREMENT)
    -   `name` (TEXT, UNIQUE)
    -   `symbols_json` (TEXT): A JSON string representing a list of `["Symbol1", "Symbol2", ...]`.
    -   `created_at` (TIMESTAMP)