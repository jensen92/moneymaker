"""本地圖像化儀表板: 用瀏覽器看權益曲線 / 月度損益 / R分布 / 交易清單.

取代純文字 Telegram 報告, 純 stdlib (http.server) + Chart.js CDN, 不需額外安裝套件
(matplotlib/flask 皆不需要; 瀏覽器透過 CDN 載入 Chart.js 自行畫圖)。

啟動 (地端測試):
    export MM_DATA_DIR=/path/to/data_adj
    python3 webapp.py                 # 預設 port 8800
    python3 webapp.py --port 9000

開啟 http://localhost:8800 ，上方輸入策略組合 (例如 C,D 或 K,L) 按「重新計算」。
資料以策略組合為 key 快取在記憶體, 同組合第二次開啟秒回。

Telegram 端可用 /chart 指令或選單按鈕取得此頁面連結 (需設定 MM_WEB_URL 環境變數,
例如內網 IP 或 ngrok 網址, 否則僅顯示 localhost 供本機開啟)。
"""
import http.server
import json
import os
import sys
import threading
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATA_DIR = os.environ.get("MM_DATA_DIR", "").strip()

_cache_lock = threading.Lock()
_cache = {"key": None, "data": None}


def _compute(keys):
    import backtest as btmod
    if DATA_DIR:
        btmod.DATA_DIR = DATA_DIR
    from backtest import (load_all, load_regime, load_regime_tiers,
                          load_vol_scalars, collect_signals, build_entry_map,
                          run_sub, INIT_CAPITAL)

    data, names = load_all()
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    init_eq = INIT_CAPITAL / len(keys)

    all_tr = []
    for k in keys:
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        entry_map = build_entry_map(sigs, data)
        trades, _ = run_sub(data, entry_map, k, 1.0, init_eq,
                            regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        all_tr.extend(trades)
    all_tr.sort(key=lambda t: t["exit_date"])

    eq = INIT_CAPITAL
    curve = []
    for t in all_tr:
        eq += t["pnl"]
        curve.append({"date": t["exit_date"].strftime("%Y-%m-%d"),
                      "equity": round(eq, 0)})

    monthly = {}
    for t in all_tr:
        mk = t["exit_date"].strftime("%Y-%m")
        monthly[mk] = monthly.get(mk, 0.0) + t["pnl"]
    monthly_list = [{"month": k, "pnl": round(v, 0)}
                    for k, v in sorted(monthly.items())]

    r_buckets = {}
    for t in all_tr:
        b = int(round(t["r"]))
        r_buckets[b] = r_buckets.get(b, 0) + 1
    r_dist = [{"r": k, "count": v} for k, v in sorted(r_buckets.items())]

    trade_list = []
    for t in all_tr[-300:][::-1]:
        trade_list.append({
            "code": t["code"], "name": names.get(t["code"], ""),
            "strategy": t["strategy"],
            "entry_date": t["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": t["exit_date"].strftime("%Y-%m-%d"),
            "entry": round(t["entry"], 2), "exit": round(t["exit"], 2),
            "r": round(t["r"], 2), "pnl": round(t["pnl"], 0),
        })

    wins = [t for t in all_tr if t["pnl"] > 0]
    n = len(all_tr)
    summary = {
        "trades": n,
        "win_rate": round(len(wins) / n, 4) if n else 0,
        "avg_r": round(sum(t["r"] for t in all_tr) / n, 3) if n else 0,
        "total_pnl": round(sum(t["pnl"] for t in all_tr), 0),
        "final_equity": round(eq, 0),
        "strategies": keys,
    }
    return {"curve": curve, "monthly": monthly_list, "r_dist": r_dist,
            "trade_list": trade_list, "summary": summary}


def get_data(strats_str):
    with _cache_lock:
        if _cache["key"] == strats_str:
            return _cache["data"]
    keys = [s.strip().upper() for s in strats_str.split(",") if s.strip()]
    result = _compute(keys)
    with _cache_lock:
        _cache["key"] = strats_str
        _cache["data"] = result
    return result


HTML_PAGE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>策略儀表板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: -apple-system, "Noto Sans TC", sans-serif; margin: 0;
         background: #0f1115; color: #e6e6e6; }
  header { padding: 14px 20px; background: #171a21; display: flex;
            align-items: center; gap: 10px; flex-wrap: wrap; }
  header input { padding: 6px 10px; border-radius: 6px; border: 1px solid #333;
                 background: #20232b; color: #fff; width: 120px; }
  header button { padding: 6px 14px; border-radius: 6px; border: none;
                   background: #3a7bd5; color: #fff; cursor: pointer; }
  .summary { display: flex; gap: 16px; padding: 14px 20px; flex-wrap: wrap; }
  .card { background: #171a21; border-radius: 10px; padding: 12px 18px; min-width: 110px; }
  .card .v { font-size: 22px; font-weight: 700; }
  .card .l { font-size: 12px; color: #9aa0aa; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 0 20px; }
  .chartbox { background: #171a21; border-radius: 10px; padding: 12px; }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #2a2d35; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  tr.win td.r { color: #4caf50; }
  tr.loss td.r { color: #ef5350; }
  .tablewrap { margin: 16px 20px; background: #171a21; border-radius: 10px;
               padding: 12px; max-height: 500px; overflow-y: auto; }
  #status { color: #9aa0aa; padding: 0 20px; font-size: 13px; }
</style>
</head>
<body>
<header>
  <strong>📊 策略儀表板</strong>
  <input id="strats" value="C,D">
  <button onclick="reload()">重新計算</button>
  <span id="status"></span>
</header>
<div class="summary" id="summary"></div>
<div class="charts">
  <div class="chartbox"><canvas id="equityChart"></canvas></div>
  <div class="chartbox"><canvas id="monthlyChart"></canvas></div>
  <div class="chartbox"><canvas id="rChart"></canvas></div>
</div>
<div class="tablewrap">
  <table id="tradeTable">
    <thead><tr><th>代號</th><th>名稱</th><th>策略</th><th>進場</th><th>出場</th>
      <th>進場價</th><th>出場價</th><th>R</th><th>損益</th></tr></thead>
    <tbody></tbody>
  </table>
</div>
<script>
let charts = {};
function destroyAll() { Object.values(charts).forEach(c => c && c.destroy()); }

function fmt(n) { return Number(n).toLocaleString(); }

async function reload() {
  const strats = document.getElementById('strats').value || 'C,D';
  const status = document.getElementById('status');
  status.textContent = '⏳ 計算中 (首次約數十秒)...';
  try {
    const r = await fetch('/api/data?strats=' + encodeURIComponent(strats));
    const j = await r.json();
    if (j.error) { status.textContent = '❌ ' + j.error; return; }
    status.textContent = '✅ 完成';
    render(j);
  } catch (e) {
    status.textContent = '❌ ' + e;
  }
}

function render(j) {
  const s = j.summary;
  document.getElementById('summary').innerHTML = `
    <div class="card"><div class="v">${s.trades}</div><div class="l">已平倉交易</div></div>
    <div class="card"><div class="v">${(s.win_rate*100).toFixed(1)}%</div><div class="l">勝率</div></div>
    <div class="card"><div class="v">${s.avg_r}</div><div class="l">均 R</div></div>
    <div class="card"><div class="v">${fmt(s.total_pnl)}</div><div class="l">總損益</div></div>
    <div class="card"><div class="v">${fmt(s.final_equity)}</div><div class="l">最終權益</div></div>
  `;

  destroyAll();
  charts.eq = new Chart(document.getElementById('equityChart'), {
    type: 'line',
    data: { labels: j.curve.map(p => p.date),
            datasets: [{ label: '權益曲線', data: j.curve.map(p => p.equity),
                         borderColor: '#3a7bd5', pointRadius: 0, tension: 0.1 }] },
    options: { plugins: { title: { display: true, text: '權益曲線', color: '#e6e6e6' },
                           legend: { display: false } },
               scales: { x: { ticks: { color: '#9aa0aa', maxTicksLimit: 10 } },
                         y: { ticks: { color: '#9aa0aa' } } } }
  });

  charts.monthly = new Chart(document.getElementById('monthlyChart'), {
    type: 'bar',
    data: { labels: j.monthly.map(p => p.month),
            datasets: [{ label: '月損益', data: j.monthly.map(p => p.pnl),
                         backgroundColor: j.monthly.map(p => p.pnl >= 0 ? '#4caf50' : '#ef5350') }] },
    options: { plugins: { title: { display: true, text: '月度損益', color: '#e6e6e6' },
                           legend: { display: false } },
               scales: { x: { ticks: { color: '#9aa0aa', maxTicksLimit: 12 } },
                         y: { ticks: { color: '#9aa0aa' } } } }
  });

  charts.r = new Chart(document.getElementById('rChart'), {
    type: 'bar',
    data: { labels: j.r_dist.map(p => p.r + 'R'),
            datasets: [{ label: '筆數', data: j.r_dist.map(p => p.count),
                         backgroundColor: j.r_dist.map(p => p.r >= 0 ? '#4caf50' : '#ef5350') }] },
    options: { plugins: { title: { display: true, text: 'R 值分布', color: '#e6e6e6' },
                           legend: { display: false } },
               scales: { x: { ticks: { color: '#9aa0aa' } },
                         y: { ticks: { color: '#9aa0aa' } } } }
  });

  const tbody = document.querySelector('#tradeTable tbody');
  tbody.innerHTML = j.trade_list.map(t => `
    <tr class="${t.pnl >= 0 ? 'win' : 'loss'}">
      <td>${t.code}</td><td>${t.name}</td><td>${t.strategy}</td>
      <td>${t.entry_date}</td><td>${t.exit_date}</td>
      <td>${t.entry}</td><td>${t.exit}</td>
      <td class="r">${t.r}</td><td class="r">${fmt(t.pnl)}</td>
    </tr>`).join('');
}

reload();
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
        elif parsed.path == "/api/data":
            strats = qs.get("strats", ["C,D"])[0]
            try:
                self._send_json(get_data(strats))
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = int(os.environ.get("MM_WEB_PORT", "8800"))
    args = sys.argv[1:]
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
    with http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"儀表板啟動: http://localhost:{port}  (區網請用本機 IP 取代 localhost)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
