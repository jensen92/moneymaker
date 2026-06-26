"""台股基本面資料 (FinMind) — 抓取 / 快取 / point-in-time 查詢 / Minervini 基本面門檻.

提供策略「第一階段篩選」所需的基本面條件 (EPS 年增、ROE) 與部位管理所需的
「下次財報發布日」，全部以 **point-in-time (PIT)** 方式提供，杜絕前視偏誤:

  財報的期末日 (Q1=3/31) 不等於發布日。台股法定截止:
    Q1 → 5/15、Q2 → 8/14、Q3 → 11/14、Q4(年報) → 隔年 3/31
  本模組一律假設「資料在法定截止日才可得」(保守 PIT 代理, 因 FinMind 未提供
  逐檔實際發布日)。回測查 t 日基本面時, 只會看到截止日 <= t 的財報, 與當時
  市場真正能取得的資訊一致。

資料來源: FinMind 免費 API, 每檔抓兩個 dataset:
  - TaiwanStockFinancialStatements: 每季 EPS (基本每股盈餘) 與單季稅後淨利
    (IncomeAfterTaxes)。
  - TaiwanStockBalanceSheet: 母公司業主權益合計 (EquityAttributableToOwnersOfParent,
    即資產負債表的股東權益, 算 ROE 用 — 注意損益表也有同名欄位但那是「淨利歸屬
    母公司」非權益, 切勿混用)。
  快取於 fundamentals/<code>.json, 可續跑。

衍生指標:
  - eps_yoy : 單季 EPS 對去年同季年增率 (前期 <=0 而本期 >0 視為轉正 → 正成長)
  - roe_ttm : 近四季稅後淨利合計 / 最新母公司業主權益 (年化 ROE 近似值)
"""
import datetime as _dt
import json
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "fundamentals")
API = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()

# 我們需要的 FinMind type 欄位
_EPS = "EPS"                                  # 損益表: 基本每股盈餘 (單季)
_NI = "IncomeAfterTaxes"                       # 損益表: 稅後淨利 (單季)
_EQ = "EquityAttributableToOwnersOfParent"     # 資產負債表: 母公司業主權益合計


# ── 法定發布截止日 (PIT 可得日) ────────────────────────────────────────────

def _avail_date(period_end):
    """財報期末日字串 'YYYY-MM-DD' → 該季財報法定可得日 (datetime.date)."""
    y, m, _ = (int(x) for x in period_end.split("-"))
    if m == 3:        # Q1
        return _dt.date(y, 5, 15)
    if m == 6:        # Q2
        return _dt.date(y, 8, 14)
    if m == 9:        # Q3
        return _dt.date(y, 11, 14)
    return _dt.date(y + 1, 3, 31)   # Q4 (年報)


def _next_period_end_after(d):
    """給定日期 d, 回傳「下一個尚未發布」財報的 (期末日, 法定可得日)."""
    cands = []
    for y in (d.year - 1, d.year, d.year + 1):
        for mm, dd in ((3, 31), (6, 30), (9, 30), (12, 31)):
            pe = f"{y}-{mm:02d}-{dd:02d}"
            av = _avail_date(pe)
            if av > d:
                cands.append((av, pe))
    cands.sort()
    return cands[0] if cands else (None, None)


# ── 抓取 / 快取 ─────────────────────────────────────────────────────────────

def _cache_path(code):
    return os.path.join(CACHE_DIR, f"{code}.json")


def fetch(code, start="2009-01-01", end=None, force=False, session=None):
    """抓取單檔財報並快取為 fundamentals/<code>.json (期末日 → {eps,ni,eq})。

    已快取且非 force 則直接讀檔。回傳 dict: {period_end: {...}} 或 {} (查無)。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(code)
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    end = end or _dt.date.today().isoformat()
    get = (session or requests).get

    def _pull(dataset, wanted):
        """抓一個 dataset, 回傳 {period_end: {key: value}} (只取 wanted 內欄位)。"""
        params = {"dataset": dataset, "data_id": code,
                  "start_date": start, "end_date": end}
        if TOKEN:
            params["token"] = TOKEN
        for attempt in range(5):
            try:
                r = get(API, params=params, timeout=30)
                if r.status_code in (402, 429):     # 限流: 退避重試
                    time.sleep(min(60, 10 * (attempt + 1)))
                    continue
                r.raise_for_status()
                j = r.json()
                if j.get("status") != 200:
                    if "limit" in str(j.get("msg", "")).lower():
                        time.sleep(min(60, 10 * (attempt + 1)))
                        continue
                    return None
                out = {}
                for x in j.get("data", []):
                    t = x.get("type")
                    if t in wanted:
                        out.setdefault(x["date"], {})[wanted[t]] = x.get("value")
                return out
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        return None

    fin = _pull("TaiwanStockFinancialStatements", {_EPS: "eps", _NI: "ni"})
    # 母公司業主權益優先, 缺漏時以權益總額 (含少數股權) 後備 → 提高 ROE 覆蓋率
    bs = _pull("TaiwanStockBalanceSheet", {_EQ: "eq", "Equity": "eq_total"})
    if fin is None and bs is None:
        # 兩個 dataset 都失敗: 寫空檔避免反覆重抓 (force 可覆寫)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}

    recs = {}
    for pe, v in (fin or {}).items():
        recs.setdefault(pe, {}).update(v)
    for pe, v in (bs or {}).items():
        recs.setdefault(pe, {}).update(v)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    return recs


def fetch_many(codes, sleep=0.3, log=print):
    """批次抓取 (跳過已快取, 可續跑)。回傳 (成功數, 空資料數)。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    sess = requests.Session()
    ok = empty = 0
    todo = [c for c in codes if not os.path.exists(_cache_path(c))]
    log(f"基本面抓取: 共 {len(codes)} 檔, 待抓 {len(todo)} 檔 (其餘已快取)")
    for n, c in enumerate(todo, 1):
        recs = fetch(c, session=sess)
        if recs:
            ok += 1
        else:
            empty += 1
        if n % 50 == 0:
            log(f"  ...{n}/{len(todo)} (有資料 {ok}, 空 {empty})")
        time.sleep(sleep)
    log(f"基本面抓取完成: 有資料 {ok}, 空 {empty}, 已快取 {len(codes) - len(todo)}")
    return ok, empty


# ── PIT 資料庫 ──────────────────────────────────────────────────────────────

class FundamentalDB:
    """讀取快取, 提供 PIT 基本面查詢與 Minervini 門檻判定。"""

    def __init__(self, codes=None):
        self.q = {}        # code -> sorted list of (avail_date, period_end, eps, ni, eq)
        if codes is None:
            if os.path.isdir(CACHE_DIR):
                codes = [f[:-5] for f in os.listdir(CACHE_DIR)
                         if f.endswith(".json")]
            else:
                codes = []
        for c in codes:
            self._load(c)

    def _load(self, code):
        path = _cache_path(code)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            recs = json.load(f)
        seq = []
        for pe, v in recs.items():
            eq = v.get("eq")
            if eq is None:
                eq = v.get("eq_total")     # 後備: 權益總額
            seq.append((_avail_date(pe), pe, v.get("eps"), v.get("ni"), eq))
        seq.sort()
        if seq:
            self.q[code] = seq

    def has(self, code):
        return code in self.q

    def _as_of(self, code, date):
        """回傳截止日 <= date 的所有財報 (由舊到新)。date 可為 date 或 Timestamp."""
        if hasattr(date, "date"):
            date = date.date()
        seq = self.q.get(code)
        if not seq:
            return []
        return [r for r in seq if r[0] <= date]

    def metrics(self, code, date):
        """PIT 計算 (eps_yoy, roe_ttm); 資料不足回傳 (None, None)。"""
        avail = self._as_of(code, date)
        if not avail:
            return None, None
        # eps_yoy: 最新一季 vs 去年同季 (期末日月份相同, 年份 -1)
        eps_yoy = None
        latest = avail[-1]
        pe = latest[1]
        cur_eps = latest[2]
        y, m, d = pe.split("-")
        prior_pe = f"{int(y) - 1}-{m}-{d}"
        prior = next((r for r in avail if r[1] == prior_pe), None)
        if cur_eps is not None and prior is not None and prior[2] is not None:
            base = prior[2]
            if base > 0:
                eps_yoy = cur_eps / base - 1.0
            elif base <= 0 < cur_eps:
                eps_yoy = 1.0      # 由虧轉盈 → 視為正成長
            else:
                eps_yoy = -1.0
        # roe_ttm: 近四季稅後淨利合計 / 最近一筆已知權益 (carry-forward, 因
        # 資產負債表權益偶有季度缺漏; 權益逐季變化小, 用最近一期為合理近似)
        roe = None
        if len(avail) >= 4:
            ttm_ni = sum(r[3] for r in avail[-4:] if r[3] is not None)
            eq = next((r[4] for r in reversed(avail) if r[4] is not None), None)
            if eq and eq > 0 and any(r[3] is not None for r in avail[-4:]):
                roe = ttm_ni / eq
        return eps_yoy, roe

    def passes(self, code, date, eps_yoy_min=0.0, roe_min=0.15,
               missing="fail"):
        """Minervini 基本面門檻: EPS 年增 >= eps_yoy_min 且 ROE >= roe_min。

        missing: 無資料時的處置 ('fail' 保守剔除 / 'pass' 放行不擋)。
        """
        eps_yoy, roe = self.metrics(code, date)
        if eps_yoy is None or roe is None:
            return missing == "pass"
        return eps_yoy >= eps_yoy_min and roe >= roe_min

    def eps_annual_growth(self, code, date):
        """O'Neil CAN SLIM 的『A』: 年度 (TTM) EPS 成長率, PIT。

        近四季 EPS 合計 vs 前四季 EPS 合計 (TTM vs 去年 TTM)。需 >= 8 季資料,
        且該八季 EPS 皆有值才計算 (否則 None)。由虧轉盈視為 +100%, 由盈轉虧 -100%。
        """
        avail = self._as_of(code, date)
        if len(avail) < 8:
            return None
        ttm_now = [r[2] for r in avail[-4:]]
        ttm_prev = [r[2] for r in avail[-8:-4]]
        if any(x is None for x in ttm_now) or any(x is None for x in ttm_prev):
            return None
        s_now, s_prev = sum(ttm_now), sum(ttm_prev)
        if s_prev > 0:
            return s_now / s_prev - 1.0
        if s_prev <= 0 < s_now:
            return 1.0
        return -1.0

    def canslim(self, code, date, c_min=0.25, a_min=0.25, roe_min=0.17,
                missing="fail"):
        """O'Neil CAN SLIM 基本面三要素 (PIT):
          C 當季 EPS 年增 >= c_min
          A 年度 (TTM) EPS 成長 >= a_min
          + ROE >= roe_min (O'Neil 的報酬品質輔助篩, 非 A 本身)
        三者皆須通過。missing: 任一資料缺漏時的處置 ('fail' 剔除 / 'pass' 放行)。
        """
        eps_yoy, roe = self.metrics(code, date)
        ann = self.eps_annual_growth(code, date)
        if eps_yoy is None or roe is None or ann is None:
            return missing == "pass"
        return eps_yoy >= c_min and ann >= a_min and roe >= roe_min

    def next_earnings(self, code, date):
        """回傳 date 之後最近的財報法定發布日 (用於財報避險); 無資料用通用排程。"""
        if hasattr(date, "date"):
            date = date.date()
        return _next_period_end_after(date)[0]


if __name__ == "__main__":
    import sys
    db = FundamentalDB(["2330"])
    if not db.has("2330"):
        print("抓取 2330 ...")
        fetch("2330")
        db = FundamentalDB(["2330"])
    for ds in ("2024-06-01", "2024-09-01", "2024-12-01", "2025-06-01"):
        d = _dt.date.fromisoformat(ds)
        eps_yoy, roe = db.metrics("2330", d)
        nxt = db.next_earnings("2330", d)
        print(f"{ds}: EPS_YoY={eps_yoy if eps_yoy is None else f'{eps_yoy:+.1%}'} "
              f"ROE_TTM={roe if roe is None else f'{roe:.1%}'} "
              f"下次財報={nxt}  通過門檻={db.passes('2330', d)}")
