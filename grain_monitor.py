"""穀物價差背景即時監控 (供 telegram_bot.py 輪詢呼叫).

對 grain_spread 的三組價差持續追蹤 z-score, 一旦:
  進場: |z| 越過 1.5 (價差偏離均值) → 推播做多/做空價差 + 兩腿掛單價 + 出場目標
  出場: 持倉中且 |z| 回到 0.3 以內 (回均值) → 推播平倉
狀態存 grain_state.json (gitignored), 跨輪詢/重啟不重複通知。

穀物為日線資料 (每日收盤後更新), 故輪詢間隔長 (預設每 1 小時) 即足夠。
回傳: (alert_text or None, state)
"""
import json
import os

import grain_spread as gx

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "grain_state.json")
_last_fetch = 0.0
FETCH_MIN = 180   # 穀物日線每 N 分鐘才重抓 (省流量)


def _load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(s):
    with open(STATE_PATH, "w") as f:
        json.dump(s, f)


def check_live(fetch=True):
    """檢查三組價差, 回傳 (alert or None, state)。fetch=True 時先更新日線。"""
    global _last_fetch
    import time
    if fetch and time.time() - _last_fetch > FETCH_MIN * 60:
        if gx.fetch_grains():
            _last_fetch = time.time()

    state = _load_state()
    alerts = []
    for a, b, name in gx.PAIRS:
        cz = gx.current_z(a, b)
        if cz is None:
            continue
        z, sp_now, mean, std, pxA, pxB, d = cz
        nA, nB = name[:2], name[3:]
        key = f"{a}{b}"
        st = state.get(key, {"pos": 0})
        pos = st.get("pos", 0)

        # 首次啟動該組: 對齊現況, 不補發 (避免一啟動就洗一排歷史訊號)
        if key not in state:
            state[key] = {"pos": 1 if z < -gx.Z_IN else (-1 if z > gx.Z_IN else 0)}
            continue

        if pos == 0:
            if z < -gx.Z_IN:       # 做多價差
                state[key] = {"pos": 1}
                alerts.append(
                    f"🟢 {name}價差 做多（多{nA}/空{nB}） z={z:+.2f}\n"
                    f"進場 買{nA}@{pxA:,.1f} 賣{nB}@{pxB:,.1f}｜價差 {sp_now:,.1f}\n"
                    f"出場目標 價差回 {mean:,.1f}（≈${(mean-sp_now)*gx.PV:,.0f}/組）")
            elif z > gx.Z_IN:      # 做空價差
                state[key] = {"pos": -1}
                alerts.append(
                    f"🔴 {name}價差 做空（空{nA}/多{nB}） z={z:+.2f}\n"
                    f"進場 賣{nA}@{pxA:,.1f} 買{nB}@{pxB:,.1f}｜價差 {sp_now:,.1f}\n"
                    f"出場目標 價差回 {mean:,.1f}（≈${(sp_now-mean)*gx.PV:,.0f}/組）")
        else:
            if abs(z) < gx.Z_OUT:  # 回均值 → 平倉
                state[key] = {"pos": 0}
                alerts.append(
                    f"⚪ {name}價差 回到均值, 平倉 z={z:+.2f}\n"
                    f"現價差 {sp_now:,.1f}（均值 {mean:,.1f}）{nA}@{pxA:,.1f} {nB}@{pxB:,.1f}")

    _save_state(state)
    return ("\n\n".join(alerts) if alerts else None), state


if __name__ == "__main__":
    a, s = check_live(fetch=False)
    print(a or f"無新訊號, 狀態: {s}")
