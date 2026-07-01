# 專案狀態索引 (封存調閱用)

> 此文件封存策略以外的內容:各功能現況、研究結論、已修 bug、決策紀錄。
> 需要細節時查此文件即可,不必翻對話。最後更新:2026-07 (session 封存)。

---

## 1. Telegram 指令總覽

| 指令 | 功能 | 對應程式 |
|---|---|---|
| `/scan` | 全市場個股掃描 (PA/PB/K/L/D) + 近觸發觀察名單 | `telegram_bot._scan_all` |
| `/picks` | 今日選股候選 (PA/PB/K/L/D 各前2) | `daily_picks.py` |
| `/ay 2330` | 個股分析 + 進出場價 (原 `/analyze`) | `telegram_bot._analyze_stock` |
| `/year` | 本年度進出清單 + 績效 | `year_job` |
| `/gold` | 黃金順勢突破 (小時K, 僅做多) | `gold_signals.py` |
| `/futures [S]` | 穀物季節 (預設S; D/M高DD不列預設) | `futures_signals.py` |
| `/grain` | 穀物個別季節進出場 (黃豆/玉米) | `grain_signals.py` |
| `/energy` (`/ng`/`/cl`) | 能源季節做多 (NG/CL) | `energy_signals.py` |
| `/cta` | 19市場CTA分散趨勢投組 | `cta_signals.py` |
| `/txf` | 台指日內 + 選擇權情緒 + 波浪結構 | `txf_strategy.live_report` |
| `/wave` | 台指波浪結構 (月/週/日/時) | `txf_wave.py` |
| `/backtest` `/chart` `/status` `/info` `/refresh` `/update` `/c` | 回測/儀表板/狀態/說明/更新/同步/ClaudeCode | 各 job |

背景監控:黃金 (`BOT_GOLD_WATCH=1` 預設開, 24h)、台指盤中 (`BOT_TXF_WATCH=1`,
09:00-13:35) + 選擇權極端情緒推播、儀表板伺服器 (`BOT_WEB=1`)。

---

## 2. 各策略現況與驗證 (策略細節在各程式的 docstring)

**個股 (生產: PA/PB/K/L/D)** — 皆 `_d_features`+`_d_signal` 的 Minervini VCP 突破變體。
- 進場:訊號日『收盤』確認突破+爆量 → 次一交易日開盤買進 (漲停跳過)。
- 出場:停損 / 跌破MA50 / +20%賣半 / 最長持有;浮盈+3R保本、站上MA50後停損跟MA50。
- **C vs D EV 對抗驗證結論**:D 較佳 (EV/年 +51R vs +36R、樣本內外皆穩) 但單筆EV差
  不顯著;C 近兩年反超、中位數高。**建議 C+D 並用** (重疊 Jaccard 僅13%, 分散效益高)。

**黃金** `gold_strategies` — 小時線順勢突破, breakout=24/atr_stop=2.5/atr_n=14/只做多。
- 抗過擬合驗證已確認為穩健最優 (walk-forward + 樣本內外 + 日線26年跨regime), **勿再調參**。
- 只做多正確:做空在黃金上連空頭年都負期望 (134筆空單總 −$76k)。

**穀物季節** `grain_signals` (`/grain`) + `futures_strategies.S_SEASON`:
- 黃豆 ZS 10月進持5月:勝率56% 中位+3.9% ✅;玉米 ZC 12月進持3月:勝率62% 中位+3.7% ✅。
- **小麥 ZW 已移除** (逐年驗證:勝率38%、中位−3.4%, 正報酬僅靠2021單一離群年)。

**能源季節** `energy_signals` (`/energy`):
- NG 9月進持3月:勝率64% 平均+13.9%/年 ✅ (波動大, 近年勝率降);
- CL 12月進持6月:勝率54% 平均+11% ✅ (報酬集中2020/21/25大年)。

**CTA 分散投組** `cta_signals` (`/cta`) — 19市場 TSMOM多空+波動目標風險平價。
- 投組Sharpe 0.18 (單市場平均0.09, 分散近兩倍) MaxDD 8.2%, 前後半皆穩。
- 純商品版 Sharpe 偏溫和;價值在與台股/黃金/穀物低相關的分散, 非單獨高報酬。

**台指** `txf_strategy` (`/txf`) — 前日高低突破日內 + 兩個輔助:
- 選擇權 P/C 未平倉比率『反向情緒』(`txo_sentiment`):高P/C(恐慌)→偏多;
  極端P/C≥138後5日+1.06%勝率66%。⚠️門檻樣本內、邊際溫和, 僅輔助。
- 波浪結構 (`txf_wave`):ZigZag轉折+多框架趨勢+費波那契;波浪標記為啟發式輔助非定論。

---

## 3. 已修 Bug 史 (供回溯)

| Bug | 檔案 | 症狀 |
|---|---|---|
| 資料更新空覆寫 | gold_signals/energy_signals/grain_signals/txf_wave fetch | API回空資料時仍覆寫清空既有歷史 → 補 `if not rows: 不覆寫` |
| 盤中誤判已最新 | download_all.task / telegram_bot._update_stock_data | 官方EOD盤中未發布, `last>=cutoff`恆真整天略過 → 改『cutoff到今天才skip』+ 同日即時價就地取代 (`_replace_last_row`) |
| 除以零崩潰 | download_all.verify_and_fill | 停牌股官方收盤價0, `/c`除零 → `if not c or c<=0: continue` |
| `/ay` NameError | telegram_bot._analyze_stock | `eff_risk`未定義, 對所有股票崩潰 → 補 `eff_risk=effective_risk(tier,vsca)` |
| `/picks` 部位不一致 | daily_picks | 用滿額RISK_PCT無市況/波動縮放/25%上限 → 改用 `suggested_position` |
| 穀物價差視窗 | grain_spread (已刪) | current_z含當日 vs backtest不含 → 已統一;整個價差策略後來刪除改個別進出場 |
| 能源持有窗 | energy_signals._state | 出場月誤報空手 → 改回測同邏輯重放 |
| P/C 月底漏抓 | txo_sentiment.fetch_pcr | d2寫死28號 → 改當月最後一天 |
| gold_monitor 多棒 | gold_monitor.check_live | 一次多根新棒只看最後根 → 追補中間棒 |
| grain_monitor 幽靈平倉 | grain_monitor (已刪) | 冷啟動依z建倉 → 改空手對齊 |

策略邏輯稽核 (5-lens workflow) 已完整跑過, 上述為確認並修復的真錯誤。

---

## 4. 研究結論封存 (已測、免重跑)

- **日線多空 (穀物/能源)**:TSMOM/Donchian/均值回歸單商品皆弱或不穩, 均 EV/年 < 季節。
  結論:單商品日線方向性無穩定edge, 季節為最佳單商品做法。
- **商品交易王者法**:巨額回報來自『多市場分散+槓桿+強趨勢時代+多資產(債/匯/股指)』,
  非單一神奇訊號。純商品趨勢分散 Sharpe~0.2 (已做成 `/cta`)。要更高需多資產版(選項B, 未做)。
- **DCA/LSI 於期貨**:不適用 (期貨會到期、需槓桿);穀物長期均值回歸年化僅~3%。
- **指標標籤**:各處『MAR』實為『總損益÷最大回撤(報酬/回撤比)』非年化;顯示已改標
  『報酬/回撤比』。`/futures` 的 MAR 才是正規年化 (CAGR/DD)。

---

## 5. 部署/環境備忘

- 終端機執行:`cp .env.example .env` 填 token → `./run.sh` 或 `python3 telegram_bot.py`。
  一個程序含 bot + 儀表板 + 三個背景監控。本機建議 `.env` 設 `BOT_AUTO_UPDATE=0`。
- Docker:`docker run --env-file .env`;`.env` 自動載入 (`_load_dotenv`)。
- 資料夾 gitignored:`data/` `data_adj/` `futures_data/` (含 basket/) `web_cache/` 等;
  快取/state 檔 (`gold_state.json` 等) 亦 gitignored。
- 分散驗證工具:`python3 season_validate.py` (季節逐年%)、`gold_daily_backtest.py` (黃金26年)。

---

## 6. 待決策/未做

- **CTA 多資產版 (選項B)**:加債券ZN/ZB、外匯6E/6J、股指ES/NQ, Sharpe預期更高 (未做)。
- **黃金做空**:已驗證負期望, 維持只做多。
- **/scan 觀察名單**:字母=突破+爆量即觸發的策略;觀察名單非買點, 突破確認才買。
