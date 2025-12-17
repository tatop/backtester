// === State ===
const state = {
    selectedSymbols: [],
    availableSymbols: [],
    weights: {},
    charts: {
        equity: null,
        drawdown: null,
        weights: null,
    },
};

// === DOM Elements ===
const elements = {
    // Navigation
    navBacktest: document.getElementById('nav-backtest'),
    navData: document.getElementById('nav-data'),
    panelBacktest: document.getElementById('panel-backtest'),
    panelData: document.getElementById('panel-data'),
    
    // Backtest form
    backtestForm: document.getElementById('backtest-form'),
    symbolInput: document.getElementById('symbol-input'),
    symbolDropdown: document.getElementById('symbol-dropdown'),
    selectedSymbols: document.getElementById('selected-symbols'),
    weightsContainer: document.getElementById('weights-container'),
    weightsSum: document.getElementById('weights-sum'),
    btnEqualWeights: document.getElementById('btn-equal-weights'),
    btnRun: document.getElementById('btn-run'),
    initialCapital: document.getElementById('initial-capital'),
    rebalanceFrequency: document.getElementById('rebalance-frequency'),
    transactionCost: document.getElementById('transaction-cost'),
    alignMethod: document.getElementById('align-method'),
    benchmark: document.getElementById('benchmark'),
    btStart: document.getElementById('bt-start'),
    btEnd: document.getElementById('bt-end'),
    
    // Results
    results: document.getElementById('results'),
    metricsGrid: document.getElementById('metrics-grid'),
    chartLegend: document.getElementById('chart-legend'),
    weightsChartContainer: document.getElementById('weights-chart-container'),
    
    // Data panel
    availableSymbolsList: document.getElementById('available-symbols-list'),
    downloadForm: document.getElementById('download-form'),
    downloadSymbols: document.getElementById('download-symbols'),
    downloadStart: document.getElementById('download-start'),
    downloadEnd: document.getElementById('download-end'),
    btnDownload: document.getElementById('btn-download'),
    downloadResults: document.getElementById('download-results'),
    
    // Toast
    toastContainer: document.getElementById('toast-container'),
};

// === API Functions ===
async function fetchSymbols() {
    try {
        const response = await fetch('/api/symbols');
        const data = await response.json();
        state.availableSymbols = data.symbols;
        renderAvailableSymbols();
    } catch (error) {
        showToast('Failed to load symbols', 'error');
    }
}

async function runBacktest(params) {
    const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Backtest failed');
    }
    
    return response.json();
}

async function downloadData(symbols, start, end) {
    const response = await fetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols, start, end }),
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Download failed');
    }
    
    return response.json();
}

// === UI Functions ===
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-message">${message}</span>`;
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function setLoading(button, loading) {
    const text = button.querySelector('.btn-text');
    const spinner = button.querySelector('.btn-spinner');
    
    button.disabled = loading;
    text.hidden = loading;
    spinner.hidden = !loading;
}

// === Symbol Selector ===
function renderSelectedSymbols() {
    elements.selectedSymbols.innerHTML = state.selectedSymbols.map(symbol => `
        <span class="symbol-tag">
            ${symbol}
            <button type="button" class="symbol-tag-remove" data-symbol="${symbol}">×</button>
        </span>
    `).join('');
    
    renderWeightsInputs();
}

function addSymbol(symbol) {
    const upperSymbol = symbol.toUpperCase().trim();
    if (upperSymbol && !state.selectedSymbols.includes(upperSymbol)) {
        state.selectedSymbols.push(upperSymbol);
        state.weights[upperSymbol] = 0;
        renderSelectedSymbols();
        updateWeightsSum();
    }
    elements.symbolInput.value = '';
    elements.symbolDropdown.classList.remove('open');
}

function removeSymbol(symbol) {
    state.selectedSymbols = state.selectedSymbols.filter(s => s !== symbol);
    delete state.weights[symbol];
    renderSelectedSymbols();
    updateWeightsSum();
}

function renderSymbolDropdown(filter = '') {
    const filtered = state.availableSymbols.filter(s => 
        s.toLowerCase().includes(filter.toLowerCase()) &&
        !state.selectedSymbols.includes(s)
    );
    
    if (filtered.length === 0 || !filter) {
        elements.symbolDropdown.classList.remove('open');
        return;
    }
    
    elements.symbolDropdown.innerHTML = filtered.slice(0, 10).map(symbol => `
        <div class="symbol-option" data-symbol="${symbol}">${symbol}</div>
    `).join('');
    
    elements.symbolDropdown.classList.add('open');
}

// === Weights ===
function renderWeightsInputs() {
    if (state.selectedSymbols.length === 0) {
        elements.weightsContainer.innerHTML = '<p class="weights-placeholder">Add symbols to configure weights</p>';
        return;
    }
    
    elements.weightsContainer.innerHTML = state.selectedSymbols.map(symbol => `
        <div class="weight-item">
            <span class="weight-label">${symbol}</span>
            <div class="input-with-suffix">
                <input type="number" 
                    class="weight-input" 
                    data-symbol="${symbol}" 
                    value="${(state.weights[symbol] * 100).toFixed(1)}"
                    min="0" 
                    max="100" 
                    step="0.1">
                <span class="input-suffix">%</span>
            </div>
        </div>
    `).join('');
}

function updateWeightsSum() {
    const sum = Object.values(state.weights).reduce((a, b) => a + b, 0);
    const sumPercent = (sum * 100).toFixed(1);
    elements.weightsSum.textContent = `Sum: ${sumPercent}%`;
    elements.weightsSum.className = 'weights-sum ' + (Math.abs(sum - 1) < 0.001 ? 'valid' : 'invalid');
}

function setEqualWeights() {
    if (state.selectedSymbols.length === 0) return;
    
    const equalWeight = 1 / state.selectedSymbols.length;
    state.selectedSymbols.forEach(symbol => {
        state.weights[symbol] = equalWeight;
    });
    renderWeightsInputs();
    updateWeightsSum();
}

// === Results & Charts ===
function renderMetrics(metrics, benchmarkMetrics = null) {
    const formatValue = (value, type) => {
        if (typeof value !== 'number' || isNaN(value)) return '—';
        
        switch (type) {
            case 'percent':
                return (value * 100).toFixed(2) + '%';
            case 'volatility':
                return value.toFixed(2) + '%';
            case 'ratio':
                return value.toFixed(2);
            case 'currency':
                return '$' + value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            default:
                return value.toString();
        }
    };
    
    const getValueClass = (value, type) => {
        if (typeof value !== 'number' || isNaN(value)) return '';
        if (type === 'drawdown') return value < 0 ? 'negative' : '';
        if (type === 'percent' || type === 'ratio') return value >= 0 ? 'positive' : 'negative';
        return '';
    };
    
    const metricsConfig = [
        { key: 'total_return', label: 'Total Return', type: 'percent' },
        { key: 'cagr', label: 'CAGR', type: 'percent' },
        { key: 'volatility', label: 'Volatility', type: 'volatility' },
        { key: 'max_drawdown', label: 'Max Drawdown', type: 'percent', valueType: 'drawdown' },
        { key: 'sharpe_ratio', label: 'Sharpe Ratio', type: 'ratio' },
    ];
    
    elements.metricsGrid.innerHTML = metricsConfig.map(config => {
        const value = metrics[config.key];
        const benchValue = benchmarkMetrics?.[config.key];
        
        return `
            <div class="metric-card">
                <div class="metric-label">${config.label}</div>
                <div class="metric-value ${getValueClass(value, config.valueType || config.type)}">
                    ${formatValue(value, config.type)}
                </div>
                ${benchValue !== undefined ? `
                    <div class="metric-subtitle">
                        Benchmark: ${formatValue(benchValue, config.type)}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

function renderCharts(data) {
    // Destroy existing charts
    Object.values(state.charts).forEach(chart => chart?.destroy());
    
    // Parse NAV data
    const navDates = data.nav_series.map(p => p.date);
    const navValues = data.nav_series.map(p => p.value);
    
    // Benchmark data
    let benchmarkDates = [];
    let benchmarkValues = [];
    if (data.benchmark_nav_series) {
        benchmarkDates = data.benchmark_nav_series.map(p => p.date);
        benchmarkValues = data.benchmark_nav_series.map(p => p.value);
    }
    
    // Calculate drawdown
    const drawdownValues = calculateDrawdown(navValues);
    
    // Chart.js defaults
    Chart.defaults.color = '#8b9dc3';
    Chart.defaults.borderColor = '#1e293b';
    Chart.defaults.font.family = "'Outfit', system-ui, sans-serif";
    
    // Equity Curve
    const equityDatasets = [
        {
            label: 'Portfolio',
            data: navValues,
            borderColor: '#22d3ee',
            backgroundColor: 'rgba(34, 211, 238, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 4,
        }
    ];
    
    if (benchmarkValues.length > 0) {
        equityDatasets.push({
            label: 'Benchmark',
            data: benchmarkValues,
            borderColor: '#f59e0b',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 4,
        });
    }
    
    // Render legend
    elements.chartLegend.innerHTML = equityDatasets.map(ds => `
        <div class="chart-legend-item">
            <span class="chart-legend-color" style="background: ${ds.borderColor}"></span>
            <span>${ds.label}</span>
        </div>
    `).join('');
    
    state.charts.equity = new Chart(document.getElementById('equity-chart'), {
        type: 'line',
        data: {
            labels: navDates,
            datasets: equityDatasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#151d2e',
                    borderColor: '#1e293b',
                    borderWidth: 1,
                    titleFont: { family: "'JetBrains Mono', monospace" },
                    bodyFont: { family: "'JetBrains Mono', monospace" },
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    grid: { display: false },
                },
                y: {
                    grid: { color: '#1e293b' },
                    ticks: {
                        callback: (value) => '$' + value.toLocaleString(),
                    },
                },
            },
        },
    });
    
    // Drawdown Chart
    state.charts.drawdown = new Chart(document.getElementById('drawdown-chart'), {
        type: 'line',
        data: {
            labels: navDates,
            datasets: [{
                label: 'Drawdown',
                data: drawdownValues,
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                borderWidth: 1,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#151d2e',
                    borderColor: '#1e293b',
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => `Drawdown: ${ctx.parsed.y.toFixed(2)}%`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    grid: { display: false },
                },
                y: {
                    max: 0,
                    grid: { color: '#1e293b' },
                    ticks: {
                        callback: (value) => value.toFixed(0) + '%',
                    },
                },
            },
        },
    });
    
    // Weights Chart (if available)
    if (data.weights_over_time && data.weights_over_time.length > 0) {
        elements.weightsChartContainer.hidden = false;
        
        const weightsDates = data.weights_over_time.map(p => p.date);
        const symbols = Object.keys(data.weights_over_time[0].weights);
        const colors = ['#22d3ee', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4'];
        
        const weightsDatasets = symbols.map((symbol, i) => ({
            label: symbol,
            data: data.weights_over_time.map(p => p.weights[symbol] * 100),
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length] + '80',
            borderWidth: 1,
            fill: true,
            tension: 0.1,
            pointRadius: 0,
        }));
        
        state.charts.weights = new Chart(document.getElementById('weights-chart'), {
            type: 'line',
            data: {
                labels: weightsDates,
                datasets: weightsDatasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 12 },
                    },
                    tooltip: {
                        backgroundColor: '#151d2e',
                        borderColor: '#1e293b',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
                        },
                    },
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'month' },
                        grid: { display: false },
                        stacked: true,
                    },
                    y: {
                        min: 0,
                        max: 100,
                        stacked: true,
                        grid: { color: '#1e293b' },
                        ticks: {
                            callback: (value) => value + '%',
                        },
                    },
                },
            },
        });
    } else {
        elements.weightsChartContainer.hidden = true;
    }
}

function calculateDrawdown(values) {
    let peak = values[0];
    return values.map(value => {
        if (value > peak) peak = value;
        return ((value - peak) / peak) * 100;
    });
}

// === Data Panel ===
function renderAvailableSymbols() {
    if (state.availableSymbols.length === 0) {
        elements.availableSymbolsList.innerHTML = '<p class="loading">No symbols available. Download some data first.</p>';
        return;
    }
    
    elements.availableSymbolsList.innerHTML = state.availableSymbols.map(symbol => `
        <span class="symbol-badge">${symbol}</span>
    `).join('');
}

// === Navigation ===
function switchPanel(panel) {
    const isBacktest = panel === 'backtest';
    
    elements.panelBacktest.hidden = !isBacktest;
    elements.panelData.hidden = isBacktest;
    
    elements.navBacktest.classList.toggle('active', isBacktest);
    elements.navData.classList.toggle('active', !isBacktest);
}

// === Event Listeners ===
function initEventListeners() {
    // Navigation
    elements.navBacktest.addEventListener('click', (e) => {
        e.preventDefault();
        switchPanel('backtest');
    });
    
    elements.navData.addEventListener('click', (e) => {
        e.preventDefault();
        switchPanel('data');
    });
    
    // Symbol input
    elements.symbolInput.addEventListener('input', (e) => {
        renderSymbolDropdown(e.target.value);
    });
    
    elements.symbolInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            addSymbol(e.target.value);
        }
    });
    
    elements.symbolDropdown.addEventListener('click', (e) => {
        if (e.target.classList.contains('symbol-option')) {
            addSymbol(e.target.dataset.symbol);
        }
    });
    
    elements.selectedSymbols.addEventListener('click', (e) => {
        if (e.target.classList.contains('symbol-tag-remove')) {
            removeSymbol(e.target.dataset.symbol);
        }
    });
    
    // Click outside to close dropdown
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.symbol-selector')) {
            elements.symbolDropdown.classList.remove('open');
        }
    });
    
    // Weights
    elements.weightsContainer.addEventListener('input', (e) => {
        if (e.target.classList.contains('weight-input')) {
            const symbol = e.target.dataset.symbol;
            state.weights[symbol] = parseFloat(e.target.value) / 100 || 0;
            updateWeightsSum();
        }
    });
    
    elements.btnEqualWeights.addEventListener('click', setEqualWeights);
    
    // Backtest form
    elements.backtestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (state.selectedSymbols.length === 0) {
            showToast('Please select at least one symbol', 'error');
            return;
        }
        
        const weightsSum = Object.values(state.weights).reduce((a, b) => a + b, 0);
        if (Math.abs(weightsSum - 1) > 0.01) {
            showToast('Weights must sum to 100%', 'error');
            return;
        }
        
        setLoading(elements.btnRun, true);
        
        try {
            const params = {
                symbols: state.selectedSymbols,
                weights: state.selectedSymbols.map(s => state.weights[s]),
                initial_capital: parseFloat(elements.initialCapital.value),
                rebalance_frequency: elements.rebalanceFrequency.value,
                transaction_cost: parseFloat(elements.transactionCost.value) / 100,
                align_method: elements.alignMethod.value,
                benchmark: elements.benchmark.value || null,
                start_date: elements.btStart.value || null,
                end_date: elements.btEnd.value || null,
            };
            
            const result = await runBacktest(params);
            
            elements.results.hidden = false;
            renderMetrics(result.metrics, result.benchmark_metrics);
            renderCharts(result);
            
            showToast('Backtest completed successfully', 'success');
            
            // Scroll to results
            elements.results.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            showToast(error.message, 'error');
        } finally {
            setLoading(elements.btnRun, false);
        }
    });
    
    // Download form
    elements.downloadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const symbolsText = elements.downloadSymbols.value.trim();
        if (!symbolsText) {
            showToast('Please enter at least one symbol', 'error');
            return;
        }
        
        const symbols = symbolsText.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
        
        setLoading(elements.btnDownload, true);
        
        try {
            const result = await downloadData(
                symbols,
                elements.downloadStart.value || null,
                elements.downloadEnd.value || null
            );
            
            // Show results
            elements.downloadResults.hidden = false;
            
            let html = '';
            if (result.successes.length > 0) {
                html += '<div class="download-success">';
                html += '<strong>Downloaded:</strong> ';
                html += result.successes.map(s => `${s.symbol} (${s.rows} rows)`).join(', ');
                html += '</div>';
            }
            
            if (Object.keys(result.errors).length > 0) {
                html += '<div class="download-error">';
                html += '<strong>Errors:</strong> ';
                html += Object.entries(result.errors).map(([sym, err]) => `${sym}: ${err}`).join(', ');
                html += '</div>';
            }
            
            elements.downloadResults.innerHTML = html;
            
            // Refresh available symbols
            await fetchSymbols();
            
            showToast('Download completed', 'success');
            
        } catch (error) {
            showToast(error.message, 'error');
        } finally {
            setLoading(elements.btnDownload, false);
        }
    });
}

// === Initialize ===
async function init() {
    await fetchSymbols();
    initEventListeners();
    
    // Set default dates for download
    const today = new Date();
    const fiveYearsAgo = new Date();
    fiveYearsAgo.setFullYear(today.getFullYear() - 5);
    
    elements.downloadEnd.value = today.toISOString().split('T')[0];
    elements.downloadStart.value = fiveYearsAgo.toISOString().split('T')[0];
}

init();

