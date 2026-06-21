"""strategies_lab.py 的向量化等價實作 (numpy 布林遮罩), 供 validate_lab2.py 快速全市場掃描。

每個 vec_* 回傳該檔成立的 bar 索引陣列 (numpy int)。訊號 dict 為常數 (各策略固定停損/
出場參數), score=該 bar 的 RS rank。validate_lab2 會以原版 strategies_lab 函式抽樣交叉
驗證一致性, 確保向量化未偏離規格。
"""
import numpy as np
import pandas as pd


def _roll_max(a, w, shift=1):
    s = pd.Series(a).rolling(w).max()
    return s.shift(shift).values


def _roll_min(a, w, shift=1):
    s = pd.Series(a).rolling(w).min()
    return s.shift(shift).values


def precompute(df):
    """抽出各策略所需的 numpy 陣列 (含向量化 rolling 量), 回傳 dict。"""
    d = {}
    for col in ("open", "high", "low", "close", "volume",
                "ma20", "ma50", "ma60", "ma150", "ma200", "vol_ma50", "atr14"):
        d[col] = df[col].values.astype(float)
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    n = len(df)
    d["n"] = n
    d["turnover"] = v * c
    d["prev_c"] = np.concatenate([[np.nan], c[:-1]])
    d["prev_o"] = np.concatenate([[np.nan], d["open"][:-1]])
    d["prev_h"] = np.concatenate([[np.nan], h[:-1]])
    d["prev_l"] = np.concatenate([[np.nan], l[:-1]])
    d["ma200_21"] = pd.Series(d["ma200"]).shift(21).values
    d["bull"] = ((d["ma50"] > d["ma150"]) & (c > d["ma150"])
                 & (d["ma200"] > d["ma200_21"]) & ~np.isnan(d["ma200"]))
    # 突破用 (不含今日)
    d["p_high20"] = _roll_max(h, 20)
    d["p_high50"] = _roll_max(h, 50)
    d["p_low20"] = _roll_min(l, 20)
    d["recent_low4"] = _roll_min(l, 4)
    d["t_high15"] = _roll_max(h, 15)
    d["t_low15"] = _roll_min(l, 15)
    # PP: 前10日「下跌日」最大量 + 下跌日數
    dn_vol = np.where(c < d["prev_c"], v, 0.0)
    d["max_dn_vol10"] = pd.Series(dn_vol).rolling(10).max().shift(1).values
    dn_day = (c < d["prev_c"]).astype(float)
    d["dn_cnt10"] = pd.Series(dn_day).rolling(10).sum().shift(1).values
    # KA: 效率比率 ER(10)
    absdiff = np.abs(np.concatenate([[np.nan], np.diff(c)]))
    er_vol = pd.Series(absdiff).rolling(10).sum().values
    er_dir = np.abs(c - pd.Series(c).shift(10).values)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["er10"] = er_dir / er_vol
    # AX: ATR 低檔 (近 [i-60..i-5] 最小)
    d["atr_prev5"] = pd.Series(d["atr14"]).shift(5).values
    d["atr_low56"] = pd.Series(d["atr14"]).rolling(56).min().shift(5).values
    # QM: 動能腿 + ADR%
    d["leg_gain"] = (pd.Series(c).shift(15) / pd.Series(c).shift(55) - 1.0).values
    adr = (h - l) / c
    d["adr20"] = pd.Series(adr).rolling(20).mean().shift(1).values
    return d


def _base_idx(d, min_i, price=10.0, turnover=50e6):
    """共用基本門檻遮罩 (價格/流動性/索引下限)。"""
    n = d["n"]
    base = np.zeros(n, dtype=bool)
    if n > min_i + 1:
        base[min_i:n - 1] = True   # 與 range(min_i, n-1) 一致 (最後一根不進場)
    return base & (d["close"] >= price) & (d["turnover"] >= turnover)


def vec_pp(d, rank):
    c = d["close"]
    near20 = np.abs(c - d["ma20"]) / d["ma20"]
    near50 = np.abs(c - d["ma50"]) / d["ma50"]
    m = (_base_idx(d, 200) & (rank >= 0.80) & (d["vol_ma50"] > 0) & d["bull"]
         & (c > d["prev_c"]) & (d["dn_cnt10"] >= 1)
         & (d["volume"] > d["max_dn_vol10"])
         & (np.minimum(near20, near50) <= 0.07))
    return np.where(m)[0]


def vec_hf(d, rank):
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    base = (_base_idx(d, 70) & (rank >= 0.85) & (d["vol_ma50"] > 0)
            & (v >= d["vol_ma50"] * 1.5))
    hit = np.zeros(d["n"], dtype=bool)
    for L in range(10, 26):
        pole_low = _roll_min(l, 40, shift=L + 1)
        pole_high = _roll_max(h, 41, shift=L)
        flag_high = _roll_max(h, L, shift=1)
        flag_low = _roll_min(l, L, shift=1)
        with np.errstate(invalid="ignore"):
            m = (base & (pole_low > 0)
                 & (pole_high / pole_low - 1.0 >= 0.90)
                 & ((flag_high - flag_low) / flag_high <= 0.25)
                 & (flag_low >= pole_high * 0.75)
                 & (c > flag_high)
                 & ((c - flag_high) / flag_high <= 0.10))
        hit |= m
    return np.where(hit)[0]


def vec_ts(d, rank):
    c, l = d["close"], d["low"]
    m = (_base_idx(d, 210) & ~np.isnan(d["ma200"]) & (c >= d["ma200"])
         & (d["recent_low4"] > d["p_low20"])
         & (l < d["p_low20"]) & (c > d["p_low20"]))
    return np.where(m)[0]


def vec_oo(d, rank):
    c = d["close"]
    m = (_base_idx(d, 210) & ~np.isnan(d["ma50"]) & (c >= d["ma50"])
         & (d["open"] < d["prev_l"]) & (c > d["prev_l"]))
    return np.where(m)[0]


def vec_vb(d, rank):
    c = d["close"]
    prev_range = d["prev_h"] - d["prev_l"]
    trigger = d["open"] + 0.5 * prev_range
    m = (_base_idx(d, 210) & (rank >= 0.70) & ~np.isnan(d["ma20"]) & (c >= d["ma20"])
         & (prev_range > 0) & (d["prev_c"] >= d["prev_o"] * 0.95)
         & (d["high"] >= trigger))
    return np.where(m)[0]


def vec_ka(d, rank):
    c = d["close"]
    m = (_base_idx(d, 210) & (rank >= 0.75) & ~np.isnan(d["ma60"]) & (c >= d["ma60"])
         & (d["vol_ma50"] > 0) & (d["er10"] >= 0.60)
         & (c > d["p_high20"]) & (d["volume"] > d["vol_ma50"] * 1.3))
    return np.where(m)[0]


def vec_ax(d, rank):
    c = d["close"]
    m = (_base_idx(d, 210) & (rank >= 0.75) & ~np.isnan(d["ma60"]) & (c >= d["ma60"])
         & ~np.isnan(d["atr14"]) & (d["atr14"] > 0) & (d["vol_ma50"] > 0)
         & (d["atr14"] > d["atr_prev5"]) & (d["atr_prev5"] <= d["atr_low56"] * 1.15)
         & (c > d["p_high50"]) & (d["volume"] > d["vol_ma50"] * 1.5))
    return np.where(m)[0]


def vec_qm(d, rank):
    c = d["close"]
    with np.errstate(invalid="ignore"):
        m = (_base_idx(d, 210) & (rank >= 0.85) & (d["vol_ma50"] > 0)
             & (d["leg_gain"] >= 0.30)
             & ((d["t_high15"] - d["t_low15"]) / d["t_high15"] <= 0.15)
             & (d["adr20"] >= 0.04)
             & (c > d["t_high15"]) & (d["volume"] > d["vol_ma50"] * 1.3))
    return np.where(m)[0]


# 各策略固定訊號 dict (與 strategies_lab.py 一致)
SIG = {
    "PP": {"minervini": True, "stop_pct": 0.08, "max_hold": 9999},
    "HF": {"stop_pct": 0.12, "trail_atr": 3.0, "gain_cap": 9.99, "max_hold": 120},
    "TS": {"stop_pct": 0.06, "target_r": 2.0, "max_hold": 8},
    "OO": {"stop_pct": 0.06, "target_r": 2.0, "max_hold": 6},
    "VB": {"stop_pct": 0.08, "trail_atr": 2.5, "gain_cap": 9.99, "max_hold": 20},
    "KA": {"stop_pct": 0.08, "trail_atr": 3.0, "gain_cap": 9.99, "max_hold": 120},
    "AX": {"stop_pct": 0.10, "trail_atr": 3.0, "gain_cap": 9.99, "max_hold": 120},
    "QM": {"stop_pct": 0.10, "trail_atr": 3.0, "gain_cap": 9.99, "max_hold": 120},
}
VEC = {"PP": vec_pp, "HF": vec_hf, "TS": vec_ts, "OO": vec_oo,
       "VB": vec_vb, "KA": vec_ka, "AX": vec_ax, "QM": vec_qm}
# 接受 RS=None (不要求 RS) 的策略: score 缺 RS 時給 0.5
RS_OPTIONAL = {"TS", "OO"}
