# HydraPredict 5

HydraPredict 5 is a quantitative research and walkforward backtesting platform. It provides an end to end pipeline for historical market data processing, technical feature engineering, machine learning model training, walkforward evaluation, risk management, and static web report generation.

The platform enforces strict separation between In-Sample (IS) training periods and Out-of-Sample (OOS) evaluation periods to prevent lookahead bias and over-fitting.

---

## Table of Contents

1. Overview
2. Key Features
3. System Architecture and Pipeline
4. Repository File Reference
5. Prerequisites and Installation
6. Usage Guide
7. Output Artifacts
8. Static Web Dashboard Publishing
9. License

---

## 1. Overview

HydraPredict 5 is designed for quantitative traders, data scientists, and algorithmic research engineers who require rigorous backtesting and validation workflows. 

Traditional backtesting systems often suffer from lookahead bias and data leakage by training models across an entire historical dataset. HydraPredict 5 solves this by implementing sequential walkforward analysis. Models are trained exclusively on historical In-Sample (IS) windows and evaluated on subsequent, unseen Out-of-Sample (OOS) windows.

The platform incorporates realistic market execution conditions, including transaction fees, slippage models, spread costs, and short-selling restrictions, producing equity curves and risk metrics that closely match live market deployment.

---

## 2. Key Features

* **Walkforward Analysis Engine**: Sequential train-on-IS and evaluate-on-OOS workflow to eliminate lookahead bias and evaluate true out-of-sample performance.
* **Execution Cost & Slippage Modeling**: Includes transaction commissions, execution delay, and slippage calculations to prevent unrealistically optimistic backtest results.
* **Feature Engineering Pipeline**: Automated feature computation for technical indicators, trend metrics, volatility measures, and cross-asset correlations.
* **Machine Learning Pipeline**: Model training, parameter tuning, signal generation, and prediction probability scoring tailored for financial time series data.
* **Risk Engine**: Portfolio position sizing, drawdown management, exposure control, and risk adjusted performance metrics.
* **Multi Format Reporting Engine**: Automatically outputs backtest results in JSON, TXT, PDF, and high-resolution PNG equity curve charts.
* **Static Dashboard Generator**: Compiles raw evaluation artifacts into an interactive static web application ready for local hosting or remote deployment.

---

## 3. System Architecture and Pipeline

The HydraPredict 5 workflow is organized into six continuous stages:

1. **Data Ingestion and Processing**: Historical price data (daily or minute timeframe bar data) is ingested from local caches or API endpoints and cleaned.
2. **Feature Generation (`spy_features.py`)**: Computes input features, moving averages, momentum indicators, volatility metrics, and target labels.
3. **Model Training (`ml_pipeline.py`)**: Trains predictive models on the defined In-Sample (IS) historical range.
4. **Walkforward Backtesting (`run_walkforward_backtest.py`)**: Evaluates trained models across Out-of-Sample (OOS) time windows, calculating trades, equity curves, drawdown, Sharpe ratio, and profit factor.
5. **Risk Engine Assessment (`risk_engine.py`)**: Applies position sizing rules, leverage controls, and portfolio constraints to the raw signals.
6. **Report & Web Generation (`build_hydrapredict5_site.py`)**: Compiles evaluation stats into structured reports (JSON, PDF, PNG) and builds a zero-dependency static web site.

---

## 4. Repository File Reference

| File / Directory | Description |
| :--- | :--- |
| `run_walkforward_backtest.py` | Main entrypoint for running historical walkforward backtest evaluations. |
| `run_full_stack.py` | Full pipeline script that executes data preparation, feature engineering, training, and backtesting in sequence. |
| `run_master_pipeline.py` | Orchestrated master pipeline runner with state persistence and execution logging. |
| `run_comparison.py` | Utility script to run comparative benchmark analysis across multiple strategies or asset pools. |
| `run_1m_backtest.py` | Specialized runner for high-frequency 1-minute intraday bar backtesting. |
| `build_hydrapredict5_site.py` | Static website compiler that converts backtest JSON reports into a hosted web dashboard. |
| `spy_features.py` | Feature extraction and indicator computation engine. |
| `ml_pipeline.py` | Machine learning model training, validation, and prediction pipeline. |
| `risk_engine.py` | Portfolio risk manager, position sizing engine, and drawdown controls. |
| `quant_stack/` | Package containing core quantitative math, backtest metrics, and order handling algorithms. |
| `deploy/` | Deployment configurations and container scripts. |
| `artifacts/` | Directory where output backtest reports, charts, and static web builds are written. |
| `Dockerfile` | Container definition for containerized backtest runs. |
| `pyproject.toml` | Python project metadata and dependency configuration. |

---

## 5. Prerequisites and Installation

### Prerequisites

* Python 3.9 or higher
* pip or uv package manager
* Docker (optional, for containerized execution)

### Option A: Local Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/greedfinanace/HYDRAPREDICT5.git
cd HYDRAPREDICT5

python -m venv venv
```

Activate the virtual environment:

* On Linux / macOS:
  ```bash
  source venv/bin/activate
  ```
* On Windows (PowerShell):
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

Install project dependencies:

```bash
pip install -e .
```

### Option B: Docker Setup

Build the Docker image:

```bash
docker build -t hydrapredict5 .
```

Run the containerized backtest:

```bash
docker run -v $(pwd)/artifacts:/app/artifacts hydrapredict5
```

---

## 6. Usage Guide

### 1. Run Historical Walkforward Backtest

Execute a walkforward backtest across selected ticker symbols and specified date ranges:

```powershell
python .\run_walkforward_backtest.py `
  source root .\tmp_daily_etf26_cache `
  source format auto `
  timeframe 1d `
  train start 2017 01 03 `
  train end 2022 12 30 `
  test start 2023 01 03 `
  test end 2023 12 29 `
  benchmark symbol SPY `
  symbols SPY GLD IEF XLV XLK `
  output root .\artifacts\hydrapredict5_run
```

Equivalent bash command for Linux or macOS:

```bash
python ./run_walkforward_backtest.py \
  source root ./tmp_daily_etf26_cache \
  source format auto \
  timeframe 1d \
  train start 2017 01 03 \
  train end 2022 12 30 \
  test start 2023 01 03 \
  test end 2023 12 29 \
  benchmark symbol SPY \
  symbols SPY GLD IEF XLV XLK \
  output root ./artifacts/hydrapredict5_run
```

### 2. Run End to End Full Stack Pipeline

To execute the entire pipeline from data ingestion to backtest reporting in a single command:

```bash
python ./run_full_stack.py
```

### 3. Run Master Pipeline with State Persistence

```bash
python ./run_master_pipeline.py
```

---

## 7. Output Artifacts

Upon completing a backtest run, the results are saved in the designated output directory (default: `artifacts/hydrapredict5_run/`):

* `walkforward_report.json`: Full structured performance metrics, trade logs, parameters, and time series data.
* `walkforward_report.txt`: Human readable text summary including CAGR, Max Drawdown, Sharpe ratio, Sortino ratio, and win rate.
* `walkforward_report.pdf`: Publication ready PDF report with financial metrics and strategy benchmark comparisons.
* `walkforward_equity_curves.png`: High resolution chart plot comparing strategy cumulative return against the SPY benchmark.

---

## 8. Static Web Dashboard Publishing

HydraPredict 5 includes a static website generator to transform backtest JSON reports into a web dashboard for hosted review.

### Step 1: Build the Static Site

Run the site builder pointing to your generated report JSON:

```powershell
python .\build_hydrapredict5_site.py `
  report json .\artifacts\hydrapredict5_run\walkforward_report.json `
  output root .\artifacts\hydrapredict5_site
```

This generates a standalone web bundle in `artifacts/hydrapredict5_site/`:
* `index.html`
* `styles.css`
* `app.js`
* `data/report.json`
* `assets/`

### Step 2: Serve the Dashboard Locally

You can serve the static site locally using Python built-in HTTP server:

```powershell
python -m http.server 8080 --directory .\artifacts\hydrapredict5_site
```

Open your web browser and navigate to:

```text
http://localhost:8080
```

---

## 9. License

This project is licensed under the MIT License. See the `LICENSE` file for full terms and conditions.
