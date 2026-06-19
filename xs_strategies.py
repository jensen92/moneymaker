"""五個「學術大膽假設」橫斷面選股策略 (cross-sectional ranked portfolios).

設計動機 (見 audit: 現有 A/C/D/G/H/I/J 全是「高 RS 股突破型態」的同一個因子,
報酬流高度相關、同生共死)。這裡刻意引入 5 個來自學術文獻、彼此正交的報酬來源,
每個都是「連續分數 → 排名 → 持有前 N 名」的投組型策略 (非門檻觸發), 以避開現有
策略「訊號稀少、靠少數右尾」的脆弱性。

每個 scorer 回傳 dict{component_name: panel(date×code)}; 回測引擎在每個再平衡日
對橫斷面做 z-score 後加總各 component, 取分數最高的前 N 名等權持有。

文獻對照:
  K  Jegadeesh & Titman (1993)            — 橫斷面動能 (12-1, 跳過最近一個月避開短期反轉)
  L  Blitz, Huij & Martens (2011)         — 殘差(特質)動能 (剝離市場 beta, 動能崩盤更少)
  M  George & Hwang (2004)                — 52 週高點接近度 (錨定偏誤, 連續分數非突破事件)
  N  Frazzini & Pedersen (2014) / Baker   — 低波動 / Betting-Against-Beta (與現有書反向 tilt)
  O  Da, Gurun & Warachka (2014)          — Frog-in-the-Pan: 動能贏家中「連續緩漲」者勝「跳漲」者
"""
import numpy as np
import pandas as pd

# 形成窗 (12-1 動能): 252 交易日總窗, 跳過最近 SKIP 日 (避開 1 個月短期反轉)
FORM = 252
SKIP = 21
FORM_W = FORM - SKIP          # 形成窗有效長度 ≈ 231
BETA_W = 756                  # 殘差動能的市場模型估計窗 (≈ 36 個月)
VOL_W = 60                    # 低波動窗


def score_K(P, R, Rm, mret, ma200):
    """K — 橫斷面動能 (Jegadeesh-Titman 12-1)."""
    mom = P.shift(SKIP) / P.shift(FORM) - 1.0
    return {"mom_12_1": mom}


def score_M(P, R, Rm, mret, ma200):
    """M — 52 週高點接近度 (George-Hwang). 越接近 52 週高分越高."""
    high52 = P.rolling(FORM).max()
    return {"prox_52wh": P / high52}


def score_N(P, R, Rm, mret, ma200):
    """N — 低波動 / BAB (Frazzini-Pedersen). 波動越低分越高;
    僅在多頭結構 (close>MA200) 中取, 避免買到下跌中的低波動價值陷阱."""
    vol = R.rolling(VOL_W).std()
    score = -vol                       # 低波動 = 高分
    score = score.where(P > ma200)     # 上升結構 gate
    return {"low_vol": score}


def score_O(P, R, Rm, mret, ma200):
    """O — Frog-in-the-Pan (Da-Gurun-Warachka). 在動能贏家中, 偏好「資訊連續、
    緩步上漲」者 (多數小漲日) 而非「靠少數跳漲」者 — 有限注意力使連續資訊擴散較慢,
    後續動能更持久。score = 動能 + 平滑度 (低 information-discreteness)."""
    pret = P.shift(SKIP) / P.shift(FORM) - 1.0
    posfrac = R.gt(0).rolling(FORM_W).mean().shift(SKIP)
    negfrac = R.lt(0).rolling(FORM_W).mean().shift(SKIP)
    sign = np.sign(pret)
    info_discrete = sign * (negfrac - posfrac)      # ID: 跳漲(少數大漲日)→ 高
    smooth = -info_discrete                          # 連續緩漲 → 高分
    winners = pret.where(pret > 0)                   # 只在動能贏家中挑平滑者
    return {"mom": winners, "smooth": smooth.where(pret > 0)}


def score_L(P, R, Rm, mret, ma200):
    """L — 殘差(特質)動能 (Blitz-Huij-Martens). 以 36 個月市場模型估 alpha/beta,
    取形成窗 (12-1) 內殘差報酬的資訊比 mean(e)/std(e)。剝離市場 beta 後動能更穩、
    崩盤更少 (直接對治現有書的 beta 集中風險)."""
    var_m = mret.rolling(BETA_W).var()
    mean_m = mret.rolling(BETA_W).mean()
    mean_r = R.rolling(BETA_W).mean()
    mean_rm = Rm.rolling(BETA_W).mean()
    cov_rm = mean_rm.sub(mean_r.mul(mean_m, axis=0))
    beta = cov_rm.div(var_m, axis=0)                 # 36m beta (date×code)
    a = mean_r.sub(beta.mul(mean_m, axis=0))         # 36m alpha

    # 形成窗 (跳過最近 SKIP 日) 內的統計
    mr_f = R.rolling(FORM_W).mean().shift(SKIP)
    mm_f = mret.rolling(FORM_W).mean().shift(SKIP)
    vr_f = R.rolling(FORM_W).var().shift(SKIP)
    vm_f = mret.rolling(FORM_W).var().shift(SKIP)
    mrm_f = Rm.rolling(FORM_W).mean().shift(SKIP)
    cov_f = mrm_f.sub(mr_f.mul(mm_f, axis=0))

    mean_e = mr_f.sub(a).sub(beta.mul(mm_f, axis=0))            # 形成窗殘差均值
    var_e = (vr_f.add((beta ** 2).mul(vm_f, axis=0))
             .sub(2 * beta.mul(cov_f)))                         # 殘差變異
    std_e = np.sqrt(var_e.clip(lower=1e-12))
    return {"res_mom": mean_e / std_e}


SCORERS = {"K": score_K, "L": score_L, "M": score_M, "N": score_N, "O": score_O}

LABELS = {
    "K": "K 橫斷面動能(12-1)",
    "L": "L 殘差動能(剝離beta)",
    "M": "M 52週高點接近度",
    "N": "N 低波動/BAB",
    "O": "O Frog-in-Pan(連續緩漲)",
}
