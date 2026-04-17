# HydraPredict 5

HydraPredict 5 is a full stack historical research and execution platform with:
  automated market data ingestion and cleaning
  event/bar construction and feature pipelines
  training + walkforward backtesting
  PDF/JSON reporting
  static website publishing for hosted result review

## Quick Start

## 1) Run Historical Walkforward Backtest

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

Outputs:
  artifacts/hydrapredict5_run/walkforward_report.json
  artifacts/hydrapredict5_run/walkforward_report.txt
  artifacts/hydrapredict5_run/walkforward_report.pdf
  artifacts/hydrapredict5_run/walkforward_equity_curves.png

## 2) Build Hostable Website

```powershell
python .\build_hydrapredict5_site.py `
  report json .\artifacts\hydrapredict5_run\walkforward_report.json `
  output root .\artifacts\hydrapredict5_site
```

Outputs:
  artifacts/hydrapredict5_site/index.html
  artifacts/hydrapredict5_site/styles.css
  artifacts/hydrapredict5_site/app.js
  artifacts/hydrapredict5_site/data/report.json
  artifacts/hydrapredict5_site/assets/* (bundled report files and charts)
  artifacts/hydrapredict5_site/assets/backtest_report.diff

## 3) Serve Locally (Host Simulation)

```powershell
python -m http.server 8080 directory .\artifacts\hydrapredict5_site
```

Open:
  http://localhost:8080

## Project Entrypoints

  run_walkforward_backtest.py : historical walkforward evaluation
  run_full_stack.py : fetch/prepare/train/backtest pipeline
  run_master_pipeline.py : orchestrated master loop + state persistence
  build_hydrapredict5_site.py : static hosted report bundle
