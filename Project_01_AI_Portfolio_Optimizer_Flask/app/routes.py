# app/routes.py (Corrected for Asynchronous Live Analysis)

from flask import current_app as app, render_template, jsonify, request, send_file
from weasyprint import HTML, CSS
import io
import os
import sqlite3
import json
import joblib
from datetime import date
from celery.result import AsyncResult
import pandas as pd
import traceback

from . import data_fetcher
from . import ml_models
from . import reporting
# --- Ensure the new task is imported ---
from .tasks import run_backtest_task, run_custom_backtest_task, run_live_analysis_task
from .config import PORTFOLIOS_DB_FILE, STOCK_UNIVERSES


# In app/routes.py

# In app/routes.py, find the index function

@app.route('/')
def index():
    model_ready = os.path.exists(app.model_path)
    current_date = date.today().strftime('%Y-%m-%d')
    
    combined_universes = [{'value': name, 'name': name.replace('_', ' ')} for name in STOCK_UNIVERSES.keys()]

    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            custom_db_universes = conn.execute("SELECT id, name FROM custom_universes ORDER BY name").fetchall()
            for u in custom_db_universes:
                combined_universes.append({'value': f'custom_{u["id"]}', 'name': u['name']})
    except Exception as e:
        print(f"Could not load custom universes for dropdowns: {e}")

    # --- THIS IS THE CHANGE ---
    # Change the source to NIFTY_500
    all_stocks_for_studio = sorted(STOCK_UNIVERSES.get("NIFTY_500", []))
    # --- END OF CHANGE ---
    
    return render_template(
        'index.html', 
        model_ready=model_ready, 
        universes=combined_universes, 
        current_date=current_date,
        all_stocks_for_studio=all_stocks_for_studio
    )

# --- THIS ENTIRE ROUTE HAS BEEN REPLACED ---
@app.route('/api/analyze_and_optimize', methods=['POST'])
def analyze_and_optimize():
    """
    This route NO LONGER does the heavy lifting. It now starts the 
    background Celery task and immediately returns a task ID.
    """
    config = request.get_json()
    
    # Check if the model file exists before starting a task
    if not os.path.exists(app.model_path):
        return jsonify({'error': 'Model file not found. Please train the model.'}), 500

    task = run_live_analysis_task.delay(
        universe_name=config.get('universe', 'NIFTY_50'),
        top_n=int(config.get('top_n', 10)),
        risk_free_rate=float(config.get('risk_free', 0.06)),
        optimization_method=config.get('optimization_method', 'sharpe'),
        sector_constraints=config.get('sector_constraints') # Get constraints from request
    )
    
    return jsonify({"task_id": task.id}), 202
# --- END OF REPLACEMENT ---


# --- THIS IS THE NEW ROUTE FOR CHECKING THE ANALYSIS STATUS ---
@app.route('/api/analysis_status/<task_id>')
def analysis_status(task_id):
    """
    This route is polled by the frontend to get the status and
    final result of the live analysis task.
    """
    task_result = AsyncResult(task_id, app=app.extensions["celery"])
    response = {}
    if task_result.state == 'PENDING':
        response = {'state': task_result.state, 'status': 'Pending...'}
    elif task_result.state == 'PROGRESS':
        response = {'state': task_result.state, 'status': task_result.info.get('status', '')}
    elif task_result.state == 'SUCCESS':
        # Check if the task returned an error dictionary
        if isinstance(task_result.result, dict) and 'error' in task_result.result:
             response = {'state': 'FAILURE', 'status': task_result.result['error']}
        else:
             response = {'state': 'SUCCESS', 'result': task_result.result}
    else: # Handles FAILURE state
        response = {'state': task_result.state, 'status': str(task_result.info)}
    return jsonify(response)
# --- END OF NEW ROUTE ---


@app.route('/api/portfolios', methods=['GET'])
def get_portfolios():
    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            portfolios = cursor.execute("SELECT id, name FROM custom_portfolios ORDER BY name").fetchall()
            return jsonify([dict(p) for p in portfolios])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolios', methods=['POST'])
def save_portfolio():
    data = request.get_json()
    name = data.get('name')
    stocks = data.get('stocks')
    optimize = data.get('optimize', False)
    manual_weights = data.get('weights', {})

    if not name or not stocks:
        return jsonify({"error": "Portfolio name and stock list are required."}), 400

    weights = {}
    if optimize:
        portfolio_data = ml_models.get_portfolio_data(stocks)
        if len(portfolio_data) < 2:
            return jsonify({"error": "Need at least 2 valid stocks to optimize."}), 400
        weights = ml_models.optimize_hrp_portfolio(portfolio_data)
    else:
        total_weight = sum(manual_weights.values())
        if not (0.99 < total_weight < 1.01):
            return jsonify({"error": f"Weights must sum to 100%. Current sum: {total_weight*100:.1f}%"}), 400
        weights = {k: v for k, v in manual_weights.items() if k in stocks}

    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO custom_portfolios (name, stocks_json) VALUES (?, ?)",
                (name, json.dumps(weights))
            )
            conn.commit()
        return jsonify({"success": True, "name": name, "weights": weights}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "A portfolio with this name already exists."}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/universes', methods=['GET'])
def get_universes():
    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            universes = conn.execute("SELECT id, name FROM custom_universes ORDER BY name").fetchall()
            return jsonify([dict(u) for u in universes])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/universes', methods=['POST'])
def upload_universe():
    if 'universeFile' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['universeFile']
    name = request.form.get('universeName')

    if not name or not name.strip():
        return jsonify({"error": "Universe name is required."}), 400
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        # Use pandas to robustly read the CSV
        df = pd.read_csv(file)
        if 'Symbol' not in df.columns:
            return jsonify({"error": "CSV must contain a 'Symbol' column."}), 400
        
        # Clean up the symbols: convert to string, uppercase, and remove whitespace
        symbols = df['Symbol'].astype(str).str.strip().str.upper().unique().tolist()
        
        if not symbols:
            return jsonify({"error": "No valid symbols found in the CSV file."}), 400

        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO custom_universes (name, symbols_json) VALUES (?, ?)",
                (name.strip(), json.dumps(symbols))
            )
            conn.commit()
        return jsonify({"success": True, "name": name, "count": len(symbols)}), 201

    except sqlite3.IntegrityError:
        return jsonify({"error": f"A universe with the name '{name}' already exists."}), 409
    except Exception as e:
        print(f"Universe Upload Error: {e}")
        return jsonify({"error": "Failed to parse or save the CSV file."}), 500
@app.route('/api/run_backtest', methods=['POST'])
def start_backtest():
    config = request.get_json()
    backtest_type = config.get('type', 'ml_strategy')
    risk_free_rate = float(config.get('risk_free', 0.06)) 

    if backtest_type == 'custom':
        portfolio_id = config.get('portfolio_id')
        if not portfolio_id:
            return jsonify({"error": "Portfolio ID is required for custom backtest."}), 400
        
        task = run_custom_backtest_task.delay(
            portfolio_id=portfolio_id,
            start_date_str=config.get('start_date'),
            end_date_str=config.get('end_date'),
            risk_free_rate=risk_free_rate,
            universe_name=config.get('universe') # Pass the universe name
        )
    else: 
        task = run_backtest_task.delay(
            start_date_str=config.get('start_date'),
            end_date_str=config.get('end_date'),
            universe_name=config.get('universe'),
            top_n=int(config.get('top_n', 10)),
            risk_free_rate=risk_free_rate,
            optimization_method=config.get('optimization_method', 'hrp'),
            sector_constraints=config.get('sector_constraints')
        )
        
    return jsonify({"task_id": task.id}), 202

@app.route('/api/backtest_status/<task_id>')
def backtest_status(task_id):
    task_result = AsyncResult(task_id, app=app.extensions["celery"])
    response = {}
    if task_result.state == 'PENDING':
        response = {'state': task_result.state, 'status': 'Pending...'}
    elif task_result.state == 'PROGRESS':
        response = {'state': task_result.state, 'status': task_result.info.get('status', '')}
    elif task_result.state == 'SUCCESS':
        response = {'state': 'SUCCESS', 'result': task_result.result}
    else: 
        response = {'state': task_result.state, 'status': str(task_result.info)}
    return jsonify(response)

@app.route('/api/explain_factors', methods=['POST'])
def explain_factors():
    explanation = reporting.generate_factor_explanation()
    html_explanation = explanation.replace('\n\n', '<p>').replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>')
    return jsonify({'explanation': html_explanation})

@app.route('/api/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        html_content = request.get_data(as_text=True)
        if not html_content:
            return jsonify({"error": "No HTML content received."}), 400

        pdf_bytes = io.BytesIO()
        css = CSS(string='''
            @page { size: A4; margin: 1cm; }
            body { font-family: sans-serif; }
            .card { border: 1px solid #ccc; margin-bottom: 20px; page-break-inside: avoid; }
            h1, h2, h3, h4, h5 { page-break-after: avoid; }
            .js-plotly-plot { width: 100% !important; }
            .table-responsive { overflow: hidden; }
        ''')
        
        HTML(string=html_content).write_pdf(pdf_bytes, stylesheets=[css])
        pdf_bytes.seek(0)

        return send_file(
            pdf_bytes,
            as_attachment=True,
            download_name='backtest_report.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return jsonify({"error": "Failed to generate PDF."}), 500

@app.route('/api/ask_chatbot', methods=['POST'])
def ask_chatbot():
    data = request.get_json()
    question = data.get('question')
    context = data.get('context')

    if not question or not context:
        return jsonify({"answer": "Error: Missing question or context."}), 400

    raw_answer = reporting.answer_user_question(question, context)

    html_answer = raw_answer.replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>')

    return jsonify({'answer': html_answer})

# Add this new route for the benchmark chart
@app.route('/api/benchmark_composition')
def benchmark_composition():
    # NOTE: These are approximate weights for visualization and may not be current.
    nifty50_weights = {
        'RELIANCE': 10.5, 'HDFCBANK': 8.5, 'ICICIBANK': 7.5, 'INFY': 6.0,
        'TCS': 5.0, 'ITC': 4.5, 'L&T': 4.0, 'HINDUNILVR': 3.5, 'AXISBANK': 3.0,
        'KOTAKBANK': 2.8, 'BHARTIARTL': 2.7, 'BAJFINANCE': 2.5, 'SBIN': 2.3,
        'ASIANPAINT': 1.8, 'M&M': 1.7, 'HCLTECH': 1.6, 'SUNPHARMA': 1.5,
        'MARUTI': 1.4, 'TITAN': 1.3, 'ULTRACEMCO': 1.2,
        'Others': 21.7
    }
    return jsonify(nifty50_weights)
# In app/routes.py, add these new functions at the end of the file

@app.route('/api/portfolios/<int:portfolio_id>', methods=['DELETE'])
def delete_portfolio(portfolio_id):
    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM custom_portfolios WHERE id = ?", (portfolio_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({"error": "Portfolio not found."}), 404
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolios/<int:portfolio_id>', methods=['PUT'])
def rename_portfolio(portfolio_id):
    data = request.get_json()
    new_name = data.get('name')
    if not new_name:
        return jsonify({"error": "New name is required."}), 400
    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE custom_portfolios SET name = ? WHERE id = ?", (new_name, portfolio_id))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({"error": "Portfolio not found."}), 404
        return jsonify({"success": True, "name": new_name}), 200
    except sqlite3.IntegrityError:
        return jsonify({"error": "A portfolio with this name already exists."}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/universes/<int:universe_id>', methods=['DELETE'])
def delete_universe(universe_id):
    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM custom_universes WHERE id = ?", (universe_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({"error": "Universe not found."}), 404
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/universes/<int:universe_id>', methods=['PUT'])
def rename_universe(universe_id):
    data = request.get_json()
    new_name = data.get('name')
    if not new_name:
        return jsonify({"error": "New name is required."}), 400
    try:
        with sqlite3.connect(PORTFOLIOS_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE custom_universes SET name = ? WHERE id = ?", (new_name, universe_id))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({"error": "Universe not found."}), 404
        return jsonify({"success": True, "name": new_name}), 200
    except sqlite3.IntegrityError:
        return jsonify({"error": "A universe with this name already exists."}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# In app/routes.py

@app.route('/api/universe_stocks/<path:universe_name>')
def get_universe_stocks(universe_name):
    """
    Returns the list of stock symbols for a given universe name.
    This supports both static (e.g., NIFTY_50) and custom (e.g., custom_1) universes.
    """
    try:
        # The existing get_stock_universe function already does all the heavy lifting
        stocks = data_fetcher.get_stock_universe(universe_name)
        
        if not stocks:
            return jsonify({"error": f"Universe '{universe_name}' not found or is empty."}), 404
        
        # Return the list of symbols, sorted alphabetically, as a JSON array
        return jsonify(sorted(stocks))
    except Exception as e:
        print(f"Error fetching stocks for universe {universe_name}: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
# In app/routes.py, add this new function

@app.route('/api/analyze_clusters', methods=['POST'])
def analyze_clusters():
    """
    Analyzes the clusters of a given portfolio of stocks.
    """
    stocks = request.get_json().get('stocks')
    if not stocks or len(stocks) < 2:
        return jsonify({"error": "At least two stocks are required for cluster analysis."}), 400

    try:
        # 1. Get the necessary historical data
        portfolio_data = ml_models.get_portfolio_data(stocks)
        
        # 2. Perform the cluster analysis
        cluster_results = ml_models.analyze_portfolio_clusters(portfolio_data)
        if cluster_results.get("error"):
            return jsonify(cluster_results), 400
            
        # 3. Generate the AI recommendation
        ai_report = reporting.generate_cluster_elimination_report(
            cluster_results['tight_clusters'],
            cluster_results['main_clusters'],
            cluster_results['feature_data']
        )
        
        return jsonify({
            "dendrogram_json": cluster_results['dendrogram_json'],
            "ai_report": ai_report
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500       