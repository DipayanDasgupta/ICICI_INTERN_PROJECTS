  
    async function loadCustomUniverses() {
        try {
            const response = await fetch('/api/universes');
            const universes = await response.json();
            const container = document.getElementById('savedUniversesContainer');

            if (container) {
                if (universes && universes.length > 0) {
                    let listHtml = '<ul class="list-group">';
                    universes.forEach(u => {
                        listHtml += `
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                <span>${u.name}</span>
                                <div>
                                    <button class="btn btn-sm btn-outline-primary me-2" onclick="renameItem('universe', ${u.id}, '${u.name}')">Rename</button>
                                    <button class="btn btn-sm btn-outline-danger" onclick="deleteItem('universe', ${u.id})">Delete</button>
                                </div>
                            </li>`;
                    });
                    listHtml += '</ul>';
                    container.innerHTML = listHtml;
                } else {
                    container.innerHTML = '<p class="text-muted">No custom universes saved yet.</p>';
                }
            }
        } catch (e) {
            console.error("Failed to load custom universes", e);
        }
        
    }

    async function loadCustomPortfolios() {
        try {
            const selector = document.getElementById('customPortfolioSelector');
            const container = document.getElementById('savedPortfoliosContainer');
            const response = await fetch('/api/portfolios');
            const portfolios = await response.json();

            // Update the dropdown selector in the backtesting tab
            if (selector) {
                selector.innerHTML = '';
                if (portfolios && portfolios.length > 0) {
                    portfolios.forEach(p => {
                        selector.innerHTML += `<option value="${p.id}">${p.name}</option>`;
                    });
                } else {
                    selector.innerHTML = '<option disabled>No custom portfolios saved yet.</option>';
                }
            }

            // Update the interactive list in the Portfolio Studio tab
            if (container) {
                if (portfolios && portfolios.length > 0) {
                    let listHtml = '<ul class="list-group">';
                    portfolios.forEach(p => {
                        listHtml += `
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                <span>${p.name}</span>
                                <div>
                                    <button class="btn btn-sm btn-outline-primary me-2" onclick="renameItem('portfolio', ${p.id}, '${p.name}')">Rename</button>
                                    <button class="btn btn-sm btn-outline-danger" onclick="deleteItem('portfolio', ${p.id})">Delete</button>
                                </div>
                            </li>`;
                    });
                    listHtml += '</ul>';
                    container.innerHTML = listHtml;
                } else {
                    container.innerHTML = '<p class="text-muted">Your saved portfolios will appear here.</p>';
                }
            }
        } catch (e) {
            console.error("Failed to load portfolios", e);
        }
    }

  // --- NEW HELPER FUNCTIONS ---
    async function deleteItem(type, id) {
        const itemName = type === 'portfolio' ? 'portfolio' : 'universe';
        if (!confirm(`Are you sure you want to delete this ${itemName}?`)) {
            return;
        }
        try {
            const response = await fetch(`/api/${itemName}s/${id}`, { method: 'DELETE' });
            if (response.ok) {
                alert(`${itemName.charAt(0).toUpperCase() + itemName.slice(1)} deleted successfully.`);
                // Reload the relevant list
                if (type === 'portfolio') {
                    loadCustomPortfolios();
                } else {
                    loadCustomUniverses();
                    // Also reload the main page to update dropdowns
                    location.reload(); 
                }
            } else {
                const result = await response.json();
                throw new Error(result.error || `Failed to delete ${itemName}.`);
            }
        } catch (error) {
            alert(error.message);
        }
    }

    async function renameItem(type, id, currentName) {
        const itemName = type === 'portfolio' ? 'portfolio' : 'universe';
        const newName = prompt(`Enter a new name for the ${itemName}:`, currentName);
        if (!newName || newName.trim() === '' || newName === currentName) {
            return;
        }
        try {
            const response = await fetch(`/api/${itemName}s/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName.trim() })
            });
            const result = await response.json();
            if (response.ok) {
                alert(`${itemName.charAt(0).toUpperCase() + itemName.slice(1)} renamed successfully.`);
                if (type === 'portfolio') {
                    loadCustomPortfolios();
                } else {
                    loadCustomUniverses();
                    location.reload();
                }
            } else {
                throw new Error(result.error || `Failed to rename ${itemName}.`);
            }
        } catch (error) {
            alert(error.message);
        }
    }


document.addEventListener('DOMContentLoaded', function() {
    
    // --- COMMON VARIABLES & INITIALIZATION ---
    let currentBacktestResults = null;
    let pollingInterval;
    let lastBacktestLogs = [];

    // --- LIVE ANALYSIS SECTION ---
    const analyzeClustersBtn = document.getElementById('analyzeClustersBtn');
    const runAnalysisBtn = document.getElementById('runAnalysisBtn');
    const loader = document.getElementById('loader');
    const stockPicksDiv = document.getElementById('stockPicksDiv');
    const rationaleDiv = document.getElementById('rationaleDiv');
    const portfolioPieChart = document.getElementById('portfolioPieChart');
    const sectorBarChart = document.getElementById('sectorBarChart');
    
    // === NEW FUNCTION ===
    async function analyzeClusters() {
        const modalContent = document.getElementById('clusterAnalysisContent');
         // Show loading spinner
         modalContent.innerHTML = `
            <div class="text-center p-5">
               <div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>
               <p class="mt-2 text-muted">Analyzing correlations and generating AI report...</p>
            </div>`;
    
        // Get the current list of top stocks from the UI
        const stockElements = document.querySelectorAll('#stockPicksDiv tr td');
        const stocks = Array.from(stockElements).map(td => td.textContent);

        if (stocks.length < 2) {
            modalContent.innerHTML = '<p class="text-danger">Not enough stocks in the portfolio to analyze.</p>';
            return;
        }

        try {
            const response = await fetch('/api/analyze_clusters', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stocks: stocks })
            });
            const data = await response.json();
 
            if (!response.ok) { throw new Error(data.error || 'Failed to analyze clusters.'); }

            // Build the final modal content
            modalContent.innerHTML = `
                <div class="row">
                    <div class="col-lg-7">
                        <div id="clusterDendrogramChart" style="min-height: 600px;"></div>
                    </div>
                    <div class="col-lg-5">
                        <h4>AI Redundancy Report</h4>
                       <div id="clusterAiReport" class="card card-body bg-light"></div>
                    </div>
                </div>
           `;
        
            // Render the Plotly chart
            Plotly.newPlot('clusterDendrogramChart', data.dendrogram_json.data, data.dendrogram_json.layout, {responsive: true});
        
            // Display the AI report
            const aiReportContainer = document.getElementById('clusterAiReport');
            aiReportContainer.innerHTML = data.ai_report.replace(/\n\n/g, '<p>').replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        } catch (error) {
            modalContent.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
        }
    }

    if (analyzeClustersBtn) {
        // Use 'show.bs.modal' event to trigger analysis only when the modal is opened
        const clusterModal = document.getElementById('clusterAnalysisModal');
        clusterModal.addEventListener('show.bs.modal', analyzeClusters);
    }
    function updateStockRankings(rankings) {
        const container = document.getElementById('stockRankingTable');
        if (!rankings || rankings.length === 0) {
            container.innerHTML = '<p class="text-danger small">No ranking data returned.</p>';
            return;
        }
        let tableHtml = '<table class="table table-sm table-hover"><thead><tr><th>Rank</th><th>Symbol</th><th>Model Score</th></tr></thead><tbody>';
        // Display only the top 50 to keep the list manageable
        rankings.slice(0, 50).forEach((item, index) => {
            const symbol = item[0];
            const score = item[1];
            tableHtml += `<tr><td>${index + 1}</td><td>${symbol}</td><td>${score.toFixed(4)}</td></tr>`;
        });
        tableHtml += '</tbody></table>';
        container.innerHTML = tableHtml;
    }
    // In main.js, inside DOMContentLoaded

   function generateSectorConstraintUI(stockSectorMap) {
        const container = document.getElementById('sectorConstraintsContainer');
        if (!stockSectorMap || Object.keys(stockSectorMap).length === 0) {
            container.innerHTML = '<p class="text-muted small">Could not determine sectors.</p>';
            return;
        }

        // Get a unique, sorted list of sectors from the values of the map
        const uniqueSectors = [...new Set(Object.values(stockSectorMap))].sort();

        let html = '';
        uniqueSectors.forEach(sector => {
            html += `
               <div class="input-group input-group-sm mb-2">
                   <span class="input-group-text" style="width: 150px; font-size: 0.8rem;">${sector}</span>
                   <input type="number" class="form-control sector-constraint-input" placeholder="Min %" data-sector="${sector}" data-type="min" min="0" max="100">
                   <input type="number" class="form-control sector-constraint-input" placeholder="Max %" data-sector="${sector}" data-type="max" min="0" max="100">
                </div>
           `;
       });
       container.innerHTML = html;
    }
    function updateFeatureImportances(importances) {
        const chartContainer = document.getElementById('featureImportanceChart');
        if (!importances || importances.length === 0) {
            chartContainer.innerHTML = '<p class="text-danger small">No feature importance data returned.</p>';
            return;
        }

        // The data comes sorted high-to-low, but Plotly plots horizontal bars
        // from bottom-to-top, so we need to reverse the arrays.
        const reversedImportances = [...importances].reverse();

        const labels = reversedImportances.map(item => item[0]); // Feature names
        const values = reversedImportances.map(item => item[1]); // Importance scores

        const plotData = [{
            x: values,
            y: labels,
            type: 'bar',
            orientation: 'h', // This makes it a horizontal bar chart
            marker: {
                color: 'rgba(13, 110, 253, 0.7)'
            }
        }];

        const layout = {
            title: '',
            margin: {
                l: 120, // Add left margin to prevent long feature names from being cut off
                r: 20,
                t: 20,
                b: 40
            },
            xaxis: {
                title: 'Importance Score'
            },
            yaxis: {
                // This ensures the most important feature (which is last after reversing)
                // is shown at the top of the chart.
                autorange: 'reversed'
            }
        };

        Plotly.newPlot(chartContainer, plotData, layout, {responsive: true});
    }

    if (runAnalysisBtn) { runAnalysisBtn.addEventListener('click', runFullAnalysis); }
    
    function displayFactorExposure(data) {
        const chartContainer = document.getElementById('factorExposureChart');
        const tableContainer = document.getElementById('factorExposureTable');

        if (data.error) {
            chartContainer.innerHTML = `<p class="text-danger small">${data.error}</p>`;
            tableContainer.innerHTML = '';
            return;
        }

        // 1. Create the Bar Chart for Betas
        const betas = data.betas;
        const labels = Object.keys(betas);
        const values = Object.values(betas);
        const colors = values.map(v => v >= 0 ? 'rgba(13, 110, 253, 0.7)' : 'rgba(220, 53, 69, 0.7)');

        const plotData = [{
            x: labels,
            y: values,
            type: 'bar',
            marker: { color: colors },
            text: values.map(v => v.toFixed(3)),
            textposition: 'auto'
        }];
        const layout = {
            title: 'Factor Betas',
            yaxis: { title: 'Beta', zeroline: true },
            xaxis: { tickangle: -20 },
            margin: { t: 40, b: 80, l: 50, r: 20 },
            height: 350
        };
        Plotly.newPlot(chartContainer, plotData, layout, {responsive: true});

        // 2. Create the Statistics Table
        let tableHtml = `<table class="table table-sm table-borderless">`;
        tableHtml += `
            <tr>
                <th class="ps-0">Annualized Alpha:</th>
                <td class="text-end fw-bold ${data.alpha_annualized_pct > 0 ? 'text-success' : 'text-danger'}">
                    ${data.alpha_annualized_pct.toFixed(2)}%
                </td>
            </tr>
            <tr>
                <th class="ps-0">R-Squared:</th>
                <td class="text-end">${(data.r_squared * 100).toFixed(1)}%</td>
            </tr>
        </table>`;

        tableHtml += '<table class="table table-sm table-hover"><thead><tr><th>Factor</th><th>Beta</th><th>T-Stat</th><th>P-Value</th></tr></thead><tbody>';
        for (const factor of labels) {
            const p_val = data.p_values[factor];
            // Highlight statistically significant results (p-value < 0.05)
            const significanceClass = p_val < 0.05 ? 'fw-bold' : '';
            tableHtml += `
                <tr class="${significanceClass}">
                    <td>${factor}</td>
                    <td>${data.betas[factor].toFixed(3)}</td>
                    <td>${data.t_stats[factor].toFixed(2)}</td>
                    <td>${p_val.toFixed(3)}</td>
                </tr>
            `;
        }
        tableHtml += '</tbody></table>';
        tableContainer.innerHTML = tableHtml;
    }

    function showLoader() { loader.style.display = 'block'; }
    function hideLoader() { loader.style.display = 'none'; }
    
    function clearLiveAnalysisResults() {
        stockPicksDiv.innerHTML = '<p class="text-muted">Run analysis to see results.</p>';
        rationaleDiv.innerHTML = '<p class="text-muted">Run analysis to generate the portfolio rationale.</p>';
        if (typeof Plotly !== 'undefined' && portfolioPieChart && sectorBarChart) {
            Plotly.purge(portfolioPieChart);
            Plotly.purge(sectorBarChart);
        }
        document.getElementById('analyzeClustersBtn').style.display = 'none';
    }
    
    function displayLiveAnalysisError(errorMsg) {
        stockPicksDiv.innerHTML = `<div class="error-message">${errorMsg}</div>`;
        rationaleDiv.innerHTML = `<p class="text-danger">Analysis failed.</p>`;
    }

    async function runFullAnalysis() {
        showLoader(); 
        clearLiveAnalysisResults();
        // Also clear the new tables
        document.getElementById('stockRankingTable').innerHTML = '<p class="text-muted small">Run analysis to see model rankings.</p>';
        document.getElementById('featureImportanceChart').innerHTML = ''; // Clear chart
        // === NEW LOGIC: Read sector constraints from the UI ===
        const sectorConstraints = {};
        document.querySelectorAll('.sector-constraint-input').forEach(input => {
            const sector = input.dataset.sector;
            const type = input.dataset.type;
            const value = parseFloat(input.value);

            if (!isNaN(value)) {
               if (!sectorConstraints[sector]) {
                   sectorConstraints[sector] = {};
                }
               sectorConstraints[sector][type] = value;
            }
        });
        // === END OF NEW LOGIC ===

        const config = { 
            universe: document.getElementById('universeSelector').value, 
            top_n: document.getElementById('topNInput').value, 
            risk_free: document.getElementById('riskFreeInput').value / 100, 
            optimization_method: document.querySelector('input[name="optMethod"]:checked').value,
            sector_constraints: sectorConstraints 
        };

        try {
            const response = await fetch('/api/analyze_and_optimize', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify(config) 
            });
            const data = await response.json();
            if (!response.ok || !data.task_id) { 
                throw new Error(data.error || 'Failed to start analysis task.');
            }
            // Start polling for the result
            pollAnalysisStatus(data.task_id);
        } catch (error) { 
            console.error('Error starting analysis:', error); 
            displayLiveAnalysisError(error.message); 
            hideLoader();
        }
    }

    function pollAnalysisStatus(taskId) {
        const rationaleDiv = document.getElementById('rationaleDiv');
        let polling = setInterval(async () => {
            try {
                const response = await fetch(`/api/analysis_status/${taskId}`);
                const data = await response.json();

                if (data.state === 'SUCCESS') {
                    clearInterval(polling);
                    hideLoader();
                    const results = data.result;
                    // Update all UI elements with the final results
                    updateStockPicks(results.top_stocks);
                    updateRationale(results.rationale);
                    plotPortfolioPie(results.optimal_weights);
                    plotSectorBar(results.sector_exposure);
                    updateStockRankings(results.stock_rankings);
                    updateFeatureImportances(results.feature_importances);
                    document.getElementById('analyzeClustersBtn').style.display = 'inline-block';
                    if (results.stock_sector_map) {
                    generateSectorConstraintUI(results.stock_sector_map);
                    }
                } else if (data.state === 'FAILURE') {
                    clearInterval(polling);
                    hideLoader();
                    displayLiveAnalysisError(data.status || 'An unknown error occurred.');
                } else {
                    // Update status message while waiting
                    rationaleDiv.innerHTML = `<p class="text-muted"><em>${data.status || 'Processing...'}</em></p>`;
                }
            } catch (error) {
                clearInterval(polling);
                hideLoader();
                displayLiveAnalysisError('A network error occurred while checking status.');
                console.error('Polling error:', error);
            }
        }, 3000); // Check every 3 seconds
    }

    function updateStockPicks(stocks) {
        if (!stocks || stocks.length === 0) { 
            stockPicksDiv.innerHTML = '<p class="text-danger">No stocks were returned.</p>'; 
            return; 
        }
        let tableHtml = '<table class="table table-sm table-hover"><tbody>';
        stocks.forEach(stock => { tableHtml += `<tr><td>${stock}</td></tr>`; });
        stockPicksDiv.innerHTML = tableHtml + '</tbody></table>';
    }

    function updateRationale(rationale) { 
        rationaleDiv.innerHTML = rationale || '<p class="text-muted">No rationale generated.</p>'; 
    }

    function plotPortfolioPie(weights) {
        if (!weights || Object.keys(weights).length === 0) return;
        const labels = Object.keys(weights).filter(k => weights[k] > 0);
        if (labels.length === 0) return;
        const values = labels.map(l => weights[l]);
        const data = [{ values, labels, type: 'pie', hole: .4, textinfo: 'label+percent', textposition: 'inside', automargin: true }];
        const layout = { title: '', showlegend: false, margin: { t: 10, b: 10, l: 10, r: 10 }, height: 350 };
        Plotly.newPlot(portfolioPieChart, data, layout, {responsive: true});
    }

    function plotSectorBar(exposure) {
        if (!exposure || Object.keys(exposure).length === 0) return;
        const sortedSectors = Object.entries(exposure).sort((a, b) => b[1] - a[1]);
        const labels = sortedSectors.map(s => s[0]);
        const values = sortedSectors.map(s => s[1] * 100);
        const data = [{ x: labels, y: values, type: 'bar', text: values.map(v => `${v.toFixed(1)}%`), textposition: 'auto' }];
        const layout = { title: '', yaxis: { title: 'Weight (%)' }, xaxis: { tickangle: -45 }, margin: { t: 10, b: 100, l: 50, r: 20 }, height: 300 };
        Plotly.newPlot(sectorBarChart, data, layout, {responsive: true});
    }

    // --- PORTFOLIO STUDIO SECTION ---
    const stockSelector = document.getElementById('stockSelector');
    const manualWeightsContainer = document.getElementById('manualWeightsContainer');
    const savePortfolioBtn = document.getElementById('savePortfolioBtn');
    const importCsvBtn = document.getElementById('importCsvBtn');
    const csvFileInput = document.getElementById('csvFileInput');
    // === ADD THESE LINES TO ACTIVATE THE NEW FEATURE ===
    if (studioUniverseSelector) {
        // Add a listener to update the stock list whenever the universe changes
        studioUniverseSelector.addEventListener('change', updateStudioStockList);
        // Call the function once on page load to populate the initial list
        updateStudioStockList(); 
    }
    // === END OF ADDITION ===

    if(importCsvBtn) { importCsvBtn.addEventListener('click', handleCsvImport); }
    if (document.querySelector('input[name="weightMethod"]')) {
        document.querySelectorAll('input[name="weightMethod"]').forEach(elem => {
            elem.addEventListener('change', function(event) {
                if (event.target.value === 'manual') {
                    updateManualWeightsUI();
                    manualWeightsContainer.style.display = 'block';
                } else {
                    manualWeightsContainer.style.display = 'none';
                }
            });
        });
    }
    if(stockSelector) stockSelector.addEventListener('change', updateManualWeightsUI);
    if(savePortfolioBtn) savePortfolioBtn.addEventListener('click', savePortfolio);
    // In app/static/js/main.js, inside the 'DOMContentLoaded' event listener

    async function updateStudioStockList() {
        const studioUniverseSelector = document.getElementById('studioUniverseSelector');
        const stockSelector = document.getElementById('stockSelector');
        if (!studioUniverseSelector || !stockSelector) return;

        const selectedUniverse = studioUniverseSelector.value;
        if (!selectedUniverse) return;

        // Provide a loading state for better UX
        stockSelector.innerHTML = '<option disabled>Loading stocks...</option>';

        try {
            const response = await fetch(`/api/universe_stocks/${selectedUniverse}`);
            const stocks = await response.json();

            // Clear the listbox before populating
            stockSelector.innerHTML = ''; 

            if (response.ok) {
               if (stocks && stocks.length > 0) {
                   stocks.forEach(symbol => {
                       const option = document.createElement('option');
                       option.value = symbol;
                       option.textContent = symbol;
                       stockSelector.appendChild(option);
                    });
                } else {
                    stockSelector.innerHTML = '<option disabled>No stocks found in this universe.</option>';
                }
            } else {
                // Display error from the API response
                stockSelector.innerHTML = `<option disabled>Error: ${stocks.error || 'Could not load stocks'}</option>`;
            }
        } catch (e) {
            console.error("Failed to update stock list:", e);
            stockSelector.innerHTML = '<option disabled>Error: Network or server issue.</option>';
        }

        // IMPORTANT: After updating the stock list, clear any manual weight inputs
        // that might be showing, as the selected stocks have changed.
        updateManualWeightsUI();
    }

    function handleCsvImport() {
        const file = csvFileInput.files[0];
        if (!file) {
            alert("Please select a CSV file to import.");
            return;
        }

        const reader = new FileReader();
        reader.onload = function(event) {
            try {
                const csv = event.target.result;
                const lines = csv.split('\n').filter(line => line.trim() !== '');
                const header = lines.shift().trim().toLowerCase().split(',');

                const symbolIndex = header.indexOf('symbol');
                const weightIndex = header.indexOf('weight');

                if (symbolIndex === -1 || weightIndex === -1) {
                    throw new Error('CSV must contain "Symbol" and "Weight" columns.');
                }

                const portfolio = lines.map(line => {
                    const values = line.trim().split(',');
                    return {
                        symbol: values[symbolIndex].trim().toUpperCase(),
                        weight: parseFloat(values[weightIndex])
                    };
                });
                
                populatePortfolioFromCsv(portfolio);

            } catch (e) {
                alert("Failed to parse CSV file. Error: " + e.message);
            }
        };
        reader.readAsText(file);
    }

    function populatePortfolioFromCsv(portfolio) {
        // 1. Set weighting method to manual
        document.getElementById('manualWeights').checked = true;
        manualWeightsContainer.style.display = 'block';

        // 2. Clear current selections and select stocks from CSV
        for(let i=0; i<stockSelector.options.length; i++) {
            stockSelector.options[i].selected = false;
        }
        portfolio.forEach(item => {
            const option = Array.from(stockSelector.options).find(opt => opt.value === item.symbol);
            if(option) {
                option.selected = true;
            } else {
                console.warn(`Symbol ${item.symbol} from CSV not found in stock list.`);
            }
        });

        // 3. Update the manual weights UI
        updateManualWeightsUI();
        
        // 4. Populate the weights from the CSV
        portfolio.forEach(item => {
            const input = document.querySelector(`.manual-weight-input[data-stock="${item.symbol}"]`);
            if(input) {
                input.value = item.weight.toFixed(2);
            }
        });

        // 5. Recalculate the total
        updateTotalWeight();
        alert("Portfolio imported successfully! Please review and save.");
    }

    function updateManualWeightsUI() {
        if (!stockSelector) return;
        const selectedStocks = Array.from(stockSelector.selectedOptions).map(opt => opt.value);
        let html = '<h6>Set Manual Weights (%)</h6>';
        selectedStocks.forEach(stock => {
            html += `
                <div class="input-group input-group-sm mb-2">
                    <span class="input-group-text" style="width: 120px;">${stock}</span>
                    <input type="number" class="form-control manual-weight-input" data-stock="${stock}" value="0" min="0" max="100" step="0.1">
                </div>
            `;
        });
        html += '<p class="text-end small text-muted mt-2">Total: <b id="manualWeightTotal">0.0</b>%</p>';
        manualWeightsContainer.innerHTML = html;
        document.querySelectorAll('.manual-weight-input').forEach(input => {
            input.addEventListener('input', updateTotalWeight);
        });
        updateTotalWeight();
    }

    function updateTotalWeight() {
        let total = 0;
        document.querySelectorAll('.manual-weight-input').forEach(input => {
            total += parseFloat(input.value) || 0;
        });
        const totalEl = document.getElementById('manualWeightTotal');
        if (totalEl) {
            totalEl.textContent = total.toFixed(1);
            totalEl.classList.remove('text-danger', 'text-success');
            if (Math.abs(total - 100.0) > 0.1 && total > 0) {
                totalEl.classList.add('text-danger');
            } else if (Math.abs(total - 100.0) < 0.1) {
                totalEl.classList.add('text-success');
            }
        }
    }
    
    async function savePortfolio() {
        const name = document.getElementById('portfolioName').value;
        const stocks = Array.from(stockSelector.selectedOptions).map(opt => opt.value);
        const method = document.querySelector('input[name="weightMethod"]:checked').value;
        const statusDiv = document.getElementById('portfolioSaveStatus');

        if (!name.trim()) { statusDiv.innerHTML = `<div class="alert alert-danger p-2">Portfolio name is required.</div>`; return; }
        if (stocks.length === 0) { statusDiv.innerHTML = `<div class="alert alert-danger p-2">Please select at least one stock.</div>`; return; }

        let payload = { name, stocks, optimize: method === 'hrp' };

        if (method === 'manual') {
            let weights = {};
            let total = 0;
            document.querySelectorAll('.manual-weight-input').forEach(input => {
                const weight = (parseFloat(input.value) || 0) / 100;
                weights[input.dataset.stock] = weight;
                total += weight;
            });
            if (Math.abs(total - 1.0) > 0.01) {
                statusDiv.innerHTML = `<div class="alert alert-danger p-2">Total weight must be exactly 100%.</div>`;
                return;
            }
            payload.weights = weights;
        }
        
        statusDiv.innerHTML = '<div class="text-muted">Saving...</div>';
        try {
            const response = await fetch('/api/portfolios', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
            const result = await response.json();
            if(response.ok) {
                statusDiv.innerHTML = `<div class="alert alert-success p-2">Portfolio '${result.name}' saved!</div>`;
                loadCustomPortfolios();
            } else {
                statusDiv.innerHTML = `<div class="alert alert-danger p-2">${result.error}</div>`;
            }
        } catch (e) {
            statusDiv.innerHTML = `<div class="alert alert-danger p-2">An unexpected error occurred.</div>`;
        }
    }



    // --- UNIVERSE STUDIO SECTION ---
    const universeUploadForm = document.getElementById('universeUploadForm');

    if (universeUploadForm) {
        universeUploadForm.addEventListener('submit', handleUniverseUpload);
    }

    async function handleUniverseUpload(e) {      
        e.preventDefault();
        const statusDiv = document.getElementById('universeUploadStatus');
        statusDiv.innerHTML = '<p class="text-muted">Uploading...</p>';

        const formData = new FormData();
        formData.append('universeFile', document.getElementById('csvUniverseFile').files[0]);
        formData.append('universeName', document.getElementById('universeName').value);

        try {
            const response = await fetch('/api/universes', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (response.ok) {
                statusDiv.innerHTML = `<div class="alert alert-success p-2">Successfully saved '${result.name}' with ${result.count} symbols.</div>`;
                setTimeout(() => {
                    location.reload();
                }, 1500);
            } else {
                throw new Error(result.error || 'An unknown error occurred.');
            }
        } catch (error) {
            statusDiv.innerHTML = `<div class="alert alert-danger p-2">Error: ${error.message}</div>`;
        }
    }


    // --- BACKTESTING SECTION ---
    const backtestBtn = document.getElementById('runBacktestBtn');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const explainFactorsBtn = document.getElementById('explainFactorsBtn');
    const backtestTypeSelector = document.getElementById('backtestTypeSelector');
    const showSectorBoxplotBtn = document.getElementById('showSectorBoxplotBtn'); 
    if (backtestBtn) backtestBtn.addEventListener('click', runBacktest);
    if (downloadCsvBtn) downloadCsvBtn.addEventListener('click', downloadLogsAsCsv);
    if (downloadPdfBtn) downloadPdfBtn.addEventListener('click', generatePdf);
    if (explainFactorsBtn) explainFactorsBtn.addEventListener('click', handleExplainFactors);
    if (backtestTypeSelector) {
        backtestTypeSelector.addEventListener('change', toggleBacktestOptions);
        loadCustomPortfolios();
        loadCustomUniverses();
    }
    if (showSectorBoxplotBtn) { showSectorBoxplotBtn.addEventListener('click', showSectorBoxplot); } 
    if (showFactorBoxplotBtn) { showFactorBoxplotBtn.addEventListener('click', showFactorBoxplot); } 

    function toggleBacktestOptions() {
        const type = backtestTypeSelector.value;
        document.getElementById('mlStrategyOptions').style.display = (type === 'ml_strategy') ? 'block' : 'none';
        document.getElementById('customPortfolioOptions').style.display = (type === 'custom') ? 'block' : 'none';
    }

    function createReturnsTable(tableDataJson, tableContainerId) {
        const container = document.getElementById(tableContainerId);
        if (!tableDataJson) { container.innerHTML = '<p class="text-muted small">No data.</p>'; return; }
        try {
            const tableData = JSON.parse(tableDataJson);
            let tableHtml = '<table class="table table-sm table-bordered table-hover text-center">';
            tableHtml += '<thead><tr><th>' + (tableData.index.name || 'Year') + '</th>';
            tableData.columns.forEach(col => { tableHtml += `<th>${col}</th>`; });
            tableHtml += '</tr></thead><tbody>';
            tableData.index.forEach((year, i) => {
                tableHtml += `<tr><th>${year}</th>`;
                tableData.data[i].forEach(val => {
                    const value = (val * 100).toFixed(2);
                    const colorClass = value > 0.01 ? 'text-success' : (value < -0.01 ? 'text-danger' : '');
                    tableHtml += `<td class="${colorClass}">${value}</td>`;
                });
                tableHtml += '</tr>';
            });
            container.innerHTML = tableHtml + '</tbody></table>';
        } catch(e) { console.error("Error building table:", e); container.innerHTML = '<p class="text-danger small">Error rendering table.</p>'; }
    }
    
    function createRebalanceLogTable(logs) {
        lastBacktestLogs = logs;
        const container = document.getElementById('rebalanceLogContainer');
        if (!logs || logs.length === 0) { container.innerHTML = '<p class="text-muted small">No rebalancing logs.</p>'; return; }
        let tableHtml = '<table class="table table-sm table-hover"><thead><tr><th>Date</th><th>Action</th><th>Details</th></tr></thead><tbody>';
        logs.forEach(log => {
            let detailsHtml = '';
            if (log.Action.includes('Hold Cash')) {
                detailsHtml = `<span class="text-muted">${log.Details}</span>`;
            } else {
                detailsHtml = Object.entries(log.Details).filter(([k, v]) => v > 0).sort((a,b) => b[1] - a[1]).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join('<br>');
            }
            tableHtml += `<tr><td>${log.Date}</td><td>${log.Action}</td><td>${detailsHtml}</td></tr>`;
        });
        container.innerHTML = tableHtml + '</tbody></table>';
    }
    
    function downloadLogsAsCsv() {
        if (lastBacktestLogs.length === 0) { alert("No log data to download."); return; }
        let csvContent = "data:text/csv;charset=utf-8,Date,Action,Symbol,Weight,Comment\n";
        lastBacktestLogs.forEach(log => {
            if (log.Action.includes('Hold Cash')) {
                csvContent += `${log.Date},Hold Cash,,,${log.Details.replace(/,/g, ";")}\n`;
            } else {
                Object.entries(log.Details).forEach(([stock, weight]) => {
                    if (weight > 0) csvContent += `${log.Date},Rebalanced,${stock},${weight.toFixed(4)},\n`;
                });
            }
        });
        const link = document.createElement("a");
        link.setAttribute("href", encodeURI(csvContent));
        link.setAttribute("download", "backtest_rebalancing_log.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    function createFullMetricsTable(kpis) {
        const container = document.getElementById('fullMetricsTableContainer');
        if (!kpis || Object.keys(kpis).length === 0) { container.innerHTML = '<p class="text-muted small">No metrics data.</p>'; return; }
        let tableHtml = '<table class="table table-sm table-striped"><tbody>';
        for (const [key, value] of Object.entries(kpis)) {
            let formattedValue = value;
            if (typeof value === 'number') {
                if (key.toLowerCase().includes('cagr') || key.includes('﹪') || key.includes('%') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('var')) {
                    formattedValue = (value * 100).toFixed(2) + '%';
                } else if (value > 1000) {
                    formattedValue = value.toLocaleString(undefined, {maximumFractionDigits: 0});
                } else {
                    formattedValue = value.toFixed(2);
                }
            }
            tableHtml += `<tr><th class="fw-normal small">${key}</th><td class="text-end">${formattedValue}</td></tr>`;
        }
        container.innerHTML = tableHtml + '</tbody></table>';
    }
    function createBenchmarkKpiTable(kpis) {
       const container = document.getElementById('benchmarkKpiContainer');
       const button = document.getElementById('showBenchmarkKpiBtn');
       if (!kpis || Object.keys(kpis).length === 0) { 
            container.innerHTML = '<p class="text-muted small">No benchmark metrics available.</p>'; 
            button.style.display = 'none'; // Hide button if no data
            return; 
        }
        let tableHtml = '<table class="table table-sm table-striped"><tbody>';
        for (const [key, value] of Object.entries(kpis)) {
            let formattedValue = value;
            if (typeof value === 'number') {
                if (key.toLowerCase().includes('cagr') || key.includes('﹪') || key.includes('%') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('var')) {
                   formattedValue = (value * 100).toFixed(2) + '%';
               } else if (value > 1000) {
                    formattedValue = value.toLocaleString(undefined, {maximumFractionDigits: 0});
                } else {
                    formattedValue = value.toFixed(2);
                }
            }
            tableHtml += `<tr><th class="fw-normal small">${key}</th><td class="text-end">${formattedValue}</td></tr>`;
        }
        container.innerHTML = tableHtml + '</tbody></table>';
        button.style.display = 'inline-block'; // Show button now that data is ready
    }

    function resetBacktestUI() {
        currentBacktestResults = null;
        document.getElementById('backtestResultContainer').style.display = 'none';
        document.getElementById('aiChatbotContainer').style.display = 'none';
        document.getElementById('showBenchmarkKpiBtn').style.display = 'none';
        if (typeof Plotly !== 'undefined') {
            Plotly.purge('backtestEquityChart');
            Plotly.purge('backtestDrawdownChart');
            Plotly.purge('historicalWeightsChart');
            Plotly.purge('historicalSectorsChart');
        }
        const kpiIds = ['cagrValue', 'sharpeValue', 'drawdownValue', 'calmarValue', 'betaValue', 'sortinoValue', 'varValue', 'cvarValue'];
        kpiIds.forEach(id => { if(document.getElementById(id)) document.getElementById(id).innerText = '-'; });
        document.getElementById('monthlyReturnsTable').innerHTML = '';
        document.getElementById('yearlyReturnsTable').innerHTML = '';
        document.getElementById('rebalanceLogContainer').innerHTML = '';
        document.getElementById('fullMetricsTableContainer').innerHTML = '<p class="text-muted small">Run a backtest to see the detailed metrics report.</p>';
        document.getElementById('aiReportContainer').innerHTML = '<p class="text-muted small">Run a backtest to generate an AI-powered analysis of the results.</p>';
        lastBacktestLogs = [];
    }

    function displayBacktestResults(results) {
        currentBacktestResults = results;
        const container = document.getElementById('backtestResultContainer');
        if (!results || !results.kpis) {
            document.getElementById('backtestStatus').innerHTML = `<div class="error-message">Received invalid or empty results from the backtest.</div>`;
            document.getElementById('backtestStatus').style.display = 'block';
            return;
        }

        const kpis = results.kpis;
        if (kpis.Error) {
            document.getElementById('backtestStatus').innerHTML = `<div class="error-message">${kpis.Error}</div>`;
            document.getElementById('backtestStatus').style.display = 'block';
            return;
        }

        const kpiMapping = {
            cagrValue: kpis['CAGR﹪'] !== undefined ? (kpis['CAGR﹪'] * 100).toFixed(2) : (kpis['CAGR (%)'] ? kpis['CAGR (%)'].toFixed(2) : '-'),
            sharpeValue: kpis['Sharpe'] !== undefined ? kpis['Sharpe'].toFixed(2) : '-',
            drawdownValue: kpis['Max Drawdown'] !== undefined ? (kpis['Max Drawdown'] * 100).toFixed(2) : (kpis['Max Drawdown [%]'] ? kpis['Max Drawdown [%]'].toFixed(2) : '-'),
            calmarValue: kpis['Calmar'] !== undefined ? kpis['Calmar'].toFixed(2) : '-',
            betaValue: kpis['Beta'] !== undefined ? kpis['Beta'].toFixed(2) : '-',
            sortinoValue: kpis['Sortino'] !== undefined ? kpis['Sortino'].toFixed(2) : '-',
            varValue: kpis['Daily VaR'] !== undefined ? (kpis['Daily VaR'] * 100).toFixed(2) + '%' : '-',
            cvarValue: kpis['Daily CVaR'] !== undefined ? (kpis['Daily CVaR'] * 100).toFixed(2) + '%' : '-'
        };
        Object.entries(kpiMapping).forEach(([id, value]) => {
            if(document.getElementById(id)) document.getElementById(id).innerText = value;
        });
        
        const plotConfig = {responsive: true};
        if (results.charts && results.charts.equity && results.charts.equity.data && results.charts.equity.layout) {
            Plotly.newPlot('backtestEquityChart', results.charts.equity.data, results.charts.equity.layout, plotConfig);
        }
        if (results.charts && results.charts.drawdown && results.charts.drawdown.data && results.charts.drawdown.layout) {
            Plotly.newPlot('backtestDrawdownChart', results.charts.drawdown.data, results.charts.drawdown.layout, plotConfig);
        }
        if (results.charts && results.charts.historical_weights && results.charts.historical_weights.data && results.charts.historical_weights.layout) {
            Plotly.newPlot('historicalWeightsChart', results.charts.historical_weights.data, results.charts.historical_weights.layout, plotConfig);
        }
        if (results.charts && results.charts.historical_sectors && results.charts.historical_sectors.data && results.charts.historical_sectors.layout) {
            Plotly.newPlot('historicalSectorsChart', results.charts.historical_sectors.data, results.charts.historical_sectors.layout, plotConfig);
        }
        
        if(results.tables) {
            createReturnsTable(results.tables.monthly_returns, 'monthlyReturnsTable');
            createReturnsTable(results.tables.yearly_returns, 'yearlyReturnsTable');
        }
        if(results.logs) createRebalanceLogTable(results.logs);
        if(results.kpis) createFullMetricsTable(results.kpis);
        if(results.benchmark_kpis) createBenchmarkKpiTable(results.benchmark_kpis);
        const aiReportContainer = document.getElementById('aiReportContainer');
        if (results.ai_report) {
            aiReportContainer.innerHTML = results.ai_report.replace(/\n\n/g, '<p>').replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>');
        } else {
            aiReportContainer.innerHTML = '<p class="text-muted small">AI report was not generated.</p>';
        }
        if (results.factor_exposure && !results.factor_exposure.error) {
            displayFactorExposure(results.factor_exposure);
        } else {
            const chartContainer = document.getElementById('factorExposureChart');
            const tableContainer = document.getElementById('factorExposureTable');
            chartContainer.innerHTML = ''; // Clear stale charts
            const errorMessage = results.factor_exposure ? results.factor_exposure.error : "Factor data not available.";
            tableContainer.innerHTML = `<div class="alert alert-warning p-2 small"><b>Factor Analysis Failed:</b><br>${errorMessage}</div>`;
        }

        // Show the chatbot container now that there are results
        document.getElementById('aiChatbotContainer').style.display = 'block';
        // Clear any previous chat history
        const chatDisplay = document.getElementById('chatDisplay');
        chatDisplay.innerHTML = '<div class="chat-message bot-message">Hello! Ask me about this backtest report.</div>';
        
        container.style.display = 'block';
    }

    function runBacktest() {
        const backtestStatusDiv = document.getElementById('backtestStatus');
        resetBacktestUI();
        backtestStatusDiv.innerHTML = `<div class="d-flex justify-content-center align-items-center"><div class="spinner-border text-primary" role="status"></div><strong class="ms-3">Starting backtest...</strong></div>`;
        backtestStatusDiv.style.display = 'block';
        if (pollingInterval) clearInterval(pollingInterval);
        // === NEW LOGIC: Read sector constraints from the Live Analysis tab UI ===
        const sectorConstraints = {};
        document.querySelectorAll('.sector-constraint-input').forEach(input => {
            const sector = input.dataset.sector;
            const type = input.dataset.type;
            const value = parseFloat(input.value);

            if (!isNaN(value)) {
                if (!sectorConstraints[sector]) {
                    sectorConstraints[sector] = {};
                }
                sectorConstraints[sector][type] = value;
            }
        });
        // === END OF NEW LOGIC ===
        const backtestType = document.getElementById('backtestTypeSelector').value;
        let config = {
            type: backtestType,
            start_date: document.getElementById('backtestStartDate').value,
            end_date: document.getElementById('backtestEndDate').value,
            risk_free: document.getElementById('riskFreeInput').value / 100,
            optimization_method: document.getElementById('backtestOptMethod').value,
            sector_constraints: sectorConstraints,
        };

        if (backtestType === 'custom') {
            config.portfolio_id = document.getElementById('customPortfolioSelector').value;
            config.universe = document.getElementById('customBacktestUniverse').value;
        } else {
            config.universe = document.getElementById('backtestUniverse').value;
            config.top_n = document.getElementById('topNInput').value;
        }

        fetch('/api/run_backtest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) })
        .then(response => response.ok ? response.json() : Promise.reject(response))
        .then(data => {
            if (data.task_id) {
                backtestStatusDiv.innerHTML += `<p class="text-muted small mt-3">Task ID: ${data.task_id}</p>`;
                pollTaskStatus(data.task_id);
            } else {
                backtestStatusDiv.innerHTML = `<div class="error-message">${data.error || 'Failed to start backtest task.'}</div>`;
            }
        })
        .catch(error => {
            console.error('Error starting backtest:', error);
            backtestStatusDiv.innerHTML = `<div class="error-message">Failed to start backtest task. Check server logs.</div>`;
        });
    }

    function pollTaskStatus(taskId) {
        const backtestStatusDiv = document.getElementById('backtestStatus');
        pollingInterval = setInterval(() => {
            fetch(`/api/backtest_status/${taskId}`)
            .then(response => response.json())
            .then(data => {
                if (data.state === 'SUCCESS') {
                    clearInterval(pollingInterval);
                    backtestStatusDiv.style.display = 'none';
                    displayBacktestResults(data.result);
                } else if (data.state === 'FAILURE') {
                    clearInterval(pollingInterval);
                    backtestStatusDiv.innerHTML = `<div class="error-message"><strong>Backtest Failed:</strong><br><pre>${data.status}</pre></div>`;
                } else {
                    let statusMessage = (data.state === 'PROGRESS' && data.status) ? data.status : (data.status || 'Processing...');
                    backtestStatusDiv.innerHTML = `<div class="d-flex justify-content-center align-items-center"><div class="spinner-border text-primary" role="status"></div><strong class="ms-3">${statusMessage}</strong></div><p class="text-muted small mt-3">Task ID: ${taskId}</p>`;
                }
            })
            .catch(error => {
                clearInterval(pollingInterval);
                console.error('Error polling task status:', error);
                backtestStatusDiv.innerHTML = `<div class="error-message">Error checking status.</div>`;
            });
        }, 3000);
    }

    function showSectorBoxplot() {
        if (!currentBacktestResults || !currentBacktestResults.charts || !currentBacktestResults.charts.historical_sectors) {
            alert("No backtest data available. Please run a backtest first.");
            return;
        }

        const sourceData = currentBacktestResults.charts.historical_sectors.data;

        // Transform the stacked bar data into box plot data
        const boxplotTraces = sourceData.map(trace => {
            return {
                x: trace.y, // Convert weights to percentages
                type: 'box',
                name: trace.name,
                orientation: 'h',
                boxpoints: 'false', // Show all the individual rebalance points
                hovertemplate: '<b>%{x}</b><br><br>' +
                           'Max: %{upperfence:.4f}<br>' +
                           'Q3: %{q3:.4f}<br>' +
                           'Median: %{median:.4f}<br>' +
                           'Q1: %{q1:.4f}<br>' +
                           'Min: %{lowerfence:.4f}' +
                           '<extra></extra>'
            };
        });

        const layout = {
            title: 'Distribution of Sector Weights Over Time',
            xaxis: {
                title: 'Allocation Weight (%)',
                zeroline: true,
                range: [0, 100]
            },
            yaxis: {
                autorange: 'reversed',
                automargin: true
            },
            margin: { t: 50, b: 50, l: 150, r: 20 },
            showlegend: false,
            hovermode: 'x unified'
        };

        const modalEl = document.getElementById('sectorBoxplotModal');
        const boxplotModal = new bootstrap.Modal(modalEl);
    
        modalEl.addEventListener('shown.bs.modal', function () {
            Plotly.newPlot('sectorBoxplotChart', boxplotTraces, layout, {responsive: true});
        }, { once: true });

        boxplotModal.show();
    }

    function showFactorBoxplot() {
        if (!currentBacktestResults || !currentBacktestResults.charts || !currentBacktestResults.charts.rolling_factor_betas) {
            alert("No rolling factor data available. Please run a backtest first.");
            return;
        }

        const rawData = currentBacktestResults.charts.rolling_factor_betas;
        if (rawData.error) {
            alert(`Could not generate plot: ${rawData.error}`);
            return;
        }

        const factorData = JSON.parse(rawData);
        const df = {};
        factorData.columns.forEach((col, i) => {
            df[col] = factorData.data.map(row => row[i]);
        });

        const boxplotTraces = [];
        for (const factorName in df) {
            if (factorName.toLowerCase() === 'alpha') continue;

            boxplotTraces.push({
                x: df[factorName],
                type: 'box',
                name: factorName,
                orientation: 'h',
                boxpoints: 'false',
                hovertemplate: '<b>%{x}</b><br><br>' +
                           'Max: %{upperfence:.4f}<br>' +
                           'Q3: %{q3:.4f}<br>' +
                           'Median: %{median:.4f}<br>' +
                           'Q1: %{q1:.4f}<br>' +
                           'Min: %{lowerfence:.4f}' +
                           '<extra></extra>'
            });
        }
        
        const layout = {
            title: 'Distribution of Rolling Factor Betas',
            xaxis: {
                title: 'Beta',
                zeroline: true,
            },
            yaxis: {
                autorange: 'reversed',
                automargin: true
            },
            margin: { t: 50, b: 50, l: 120, r: 20 },
            showlegend: false,
            hovermode: 'y'
        };

        const modalEl = document.getElementById('factorBoxplotModal');
        const boxplotModal = new bootstrap.Modal(modalEl);
        
        modalEl.addEventListener('shown.bs.modal', function () {
            Plotly.newPlot('factorBoxplotChart', boxplotTraces, layout, {responsive: true});
        }, { once: true });

        boxplotModal.show();
    } 

    // --- AI CHATBOT SECTION ---
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatDisplay = document.getElementById('chatDisplay');

    if (chatForm) {
        chatForm.addEventListener('submit', handleChatSubmit);
    }

    function getBacktestContext() {
        const context = {
            kpis: {},
            full_metrics: {},
            ai_summary: ''
        };

        const kpiIds = ['cagrValue', 'sharpeValue', 'drawdownValue', 'calmarValue', 'betaValue', 'sortinoValue', 'varValue', 'cvarValue'];
        kpiIds.forEach(id => {
            const el = document.getElementById(id);
            if(el) {
                const label = el.nextElementSibling.textContent;
                context.kpis[label] = el.textContent;
            }
        });

        const metricsTable = document.getElementById('fullMetricsTableContainer');
        metricsTable.querySelectorAll('tr').forEach(row => {
            const key = row.querySelector('th')?.textContent;
            const value = row.querySelector('td')?.textContent;
            if (key && value) {
                context.full_metrics[key] = value;
            }
        });

        context.ai_summary = document.getElementById('aiReportContainer').textContent;

        return context;
    }
    
    function appendChatMessage(message, sender) {
        const messageEl = document.createElement('div');
        messageEl.classList.add('chat-message', `${sender}-message`);
        messageEl.innerHTML = message;
        chatDisplay.appendChild(messageEl);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
        return messageEl;
    }

    async function handleChatSubmit(e) {
        e.preventDefault();
        const userMessage = chatInput.value.trim();
        if (!userMessage) return;

        appendChatMessage(userMessage, 'user');
        chatInput.value = '';

        const thinkingEl = appendChatMessage('<i>Assistant is thinking...</i>', 'bot');
        thinkingEl.classList.add('thinking');

        try {
            const context = getBacktestContext();
            
            const response = await fetch('/api/ask_chatbot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: userMessage,
                    context: context
                })
            });

            const data = await response.json();
            thinkingEl.remove();

            if (!response.ok) {
                throw new Error(data.answer || 'An unknown error occurred.');
            }
            
            appendChatMessage(data.answer, 'bot');

        } catch (error) {
            thinkingEl.remove();
            appendChatMessage(`<strong>Error:</strong> ${error.message}`, 'bot');
        }
    }

    async function handleExplainFactors(e) {
        e.preventDefault();
        
        factorExplanationContainer.style.display = 'block';
        factorExplanationContainer.innerHTML = `
            <div class="d-flex align-items-center">
                <strong>Generating explanation with AI...</strong>
                <div class="spinner-border ms-auto" role="status" aria-hidden="true"></div>
            </div>`;

        try {
            const response = await fetch('/api/explain_factors', { method: 'POST' });
            const data = await response.json();
            if(response.ok) {
                factorExplanationContainer.innerHTML = data.explanation;
            } else {
                throw new Error(data.error || 'Failed to fetch explanation.');
            }
        } catch (error) {
            factorExplanationContainer.innerHTML = `<p class="text-danger">${error.message}</p>`;
        }
    }
});

async function generatePdf() {
    const reportContainer = document.getElementById('backtestResultContainer');
    if (!reportContainer || reportContainer.style.display === 'none') {
        alert("Please run a backtest first to generate a report.");
        return;
    }

    const originalTitle = document.title;
    document.title = "Backtest_Report";

    const pdfBtn = document.getElementById('downloadPdfBtn');
    const csvBtn = document.getElementById('downloadCsvBtn');
    const factorBtn = document.getElementById('explainFactorsBtn');
    pdfBtn.style.display = 'none';
    csvBtn.style.display = 'none';
    if(factorBtn) factorBtn.style.display = 'none';

    const loader = document.getElementById('loader');
    loader.style.display = 'block';

    try {
        const reportHtml = reportContainer.innerHTML;

        const response = await fetch('/api/generate_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'text/plain' },
            body: reportHtml
        });

        if (!response.ok) {
            throw new Error('PDF generation failed on the server.');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'backtest_report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error("Error generating PDF:", error);
        alert("Could not generate PDF. See console for details.");
    } finally {
        pdfBtn.style.display = 'inline-block';
        csvBtn.style.display = 'inline-block';
        if(factorBtn) factorBtn.style.display = 'inline-block';
        document.title = originalTitle;
        loader.style.display = 'none';
    }
}