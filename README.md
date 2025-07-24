# AI-Powered Quantitative Tools for Investment Management

This repository is a consolidated collection of the key projects developed during my internship with the Investment Management Team at ICICI Prudential AMC. It showcases an evolutionary journey from simple sentiment analysis tools to comprehensive, AI-powered quantitative investment platforms.

## 📂 Projects Overview

This repository contains three primary, independent applications and the presentations that document their development:

1.  **[Project 1: AI Portfolio Optimizer & Backtesting Engine (Flask)](./Project_01_AI_Portfolio_Optimizer_Flask/README.md)**
    *   **Description**: A professional-grade, web-based platform for quantitative strategy development, robust backtesting, and advanced portfolio analysis. It features an asynchronous task queue (Celery & Redis) to handle computationally intensive operations without blocking the UI.
    *   **Technology**: Flask, Celery, Redis, LightGBM, QuantStats, Plotly.js.

2.  **[Project 2: AI Portfolio Agent (Streamlit)](./Project_02_AI_Portfolio_Agent_Streamlit/README.md)**
    *   **Description**: A real-time, AI-driven portfolio management assistant. This application provides live market dashboards, generates optimal portfolios on-demand, simulates trading strategies using Deep Reinforcement Learning, and offers comprehensive portfolio analysis.
    *   **Technology**: Streamlit, TensorFlow, OpenAI Gym, yfinance, Google Gemini.

3.  **[Project 3: Sectoral News Analyzer & Scraper (Flask)](./Project_03_Sectoral_News_Analyzer_Scraper/README.md)**
    *   **Description**: A Flask application focused on news aggregation and sentiment analysis. It includes a powerful, custom-built web scraping pipeline using Scrapy and Playwright to build a local news database, which is then used for AI-driven sectoral analysis.
    *   **Technology**: Flask, Scrapy, Playwright, NLTK, Google Gemini.

4.  **[Presentations](./_Presentations/)**
    *   Contains the Mid-Tenure and Final Internship reports, which provide the narrative, context, and key learnings from the development of these tools.

Please refer to the `README.md` file within each project's directory for detailed setup and usage instructions.
