# archive/ — 封存區 (研究 / 優化 / 已淘汰實驗)

此資料夾收納**非策略核心**的程式與文件：歷次參數優化掃描、研究腳本、已淘汰的
策略版本、一次性分析與驗證工具。保留供追溯，但**不屬於正式運行系統**。

正式運行系統 (留在根目錄) 只有三個策略主體 + 其引擎/資料/介面：

## 留在根目錄的「策略核心」

**台股選股策略 (Minervini SEPA / VCP 突破, 同一引擎多組參數 = C/D/B/K/L/PA/PB)**
- `strategies.py` — 所有策略定義 (共用 `_d_features` 模板+量價特徵 / `_d_signal` 門檻)
- `backtest.py` — 回測與下單引擎 (停損/移動停損/市況濾網/部位)
- `github_scan.py` — 每日全市場掃描報告 (GitHub Actions)
- `daily_picks.py` — 今日選股
- `download_all.py` — 還原股價資料管線 (Yahoo + 交易所官方比對)
- `telegram_bot.py` / `webapp.py` / `env_guard.py` — 介面與相依

**台指期日內策略 (前日高低突破)**
- `txf_strategy.py` / `txf_data.py` / `txf_backtest.py`

**穀物/黃金期貨策略**
- `futures_scan.py` / `futures_signals.py` / `futures_data.py`
  / `futures_backtest.py` / `futures_strategies.py`

## 為何封存這些

- `optimize_*.py` — 產生上述策略最終參數的優化掃描 (任務完成, 參數已寫回 strategies.py)。
- `research_*.py` / `round*_*.py` / `*_lab*.py` / `tune_*.py` / `sweep_*.py` — 探索性
  研究與已淘汰的策略點子 (A/E/F/G/H/I/J、xs、stf、chip_cost、tw_* 等)。
- `breadth_filter_test.py` / `entry_exit_test.py` / `txf_verify.py` / `txf_stop_sweep.py`
  / `analyze_*.py` / `corr_analysis.py` / `audit_overlap.py` / `doctor.py` 等 — 一次性
  分析與驗證, 非運行所需。
- `futures_gold*.py` / `futures_grain*.py` / `futures_portfolio.py` 等 — 期貨研究分支
  (正式 /futures 只用上面五支)。
