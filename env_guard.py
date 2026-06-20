"""在匯入 pandas 之前呼叫 apply(), 讓環境中「為 numpy 1.x 編譯、與 numpy 2.x 不相容」
的選用加速套件 (numexpr / bottleneck) 安靜停用, 避免 pandas 匯入時噴一大串
「A module that was compiled using NumPy 1.x ...」警告與 Traceback。

背景: numexpr / bottleneck 只是 pandas 的選用「加速器」, 不是必要相依。pandas 在它們
缺席時功能完全正常 (本專案資料量小, 速度幾乎無差)。anaconda 把 numpy 升到 2.x 但這兩個
C 擴充仍是舊版 (numpy 1.x ABI) 時, 載入就會出現 ABI 警告/錯誤。

策略:
  健康環境  → 套件可正常匯入, 不動它, 照常加速。
  壞掉環境  → 匯入即出錯 (ABI 不合) → 用 sys.modules[name]=None 擋掉, 讓 pandas 跳過。
這樣不論使用者本機環境如何, 機器人都能乾淨運作。真正的根治仍建議升級/移除這兩個套件
(見 doctor.py 與 DEPLOY.md)。
"""
import sys
import warnings


def _probe_and_neutralize(name):
    """嘗試匯入 name。

    回傳 "ok" (健康)、"absent" (未安裝, pandas 本就會跳過, 不用處理) 或
    "broken" (已安裝但匯入出錯, 多為 numpy 1.x ABI 不合 → 用 None 擋掉讓 pandas 跳過)。
    """
    if name in sys.modules:
        return "ok" if sys.modules[name] is not None else "broken"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            __import__(name)
        return "ok"
    except ModuleNotFoundError:
        return "absent"            # 根本沒裝 → 不動它, pandas 原生就會略過
    except BaseException:          # ImportError / AttributeError / 其它 ABI 例外
        sys.modules[name] = None   # 之後 `import name` 會立即 ImportError → pandas 跳過
        return "broken"


def apply(verbose=False):
    """在第一次 import pandas 之前呼叫。回傳被停用的套件清單。"""
    # 壓掉 numpy 對「以 1.x 編譯之模組」的警告 (即使套件健康也順手少噪音)
    warnings.filterwarnings("ignore", message=r".*compiled using NumPy 1\.x.*")
    warnings.filterwarnings("ignore", message=r".*Pandas requires version.*")
    disabled = []
    for name in ("numexpr", "bottleneck"):
        if _probe_and_neutralize(name) == "broken":
            disabled.append(name)
    if verbose and disabled:
        print(f"[env_guard] 已停用與 numpy 不相容的舊版加速套件: {', '.join(disabled)} "
              f"(pandas 仍可正常運作; 根治方式請執行 python3 doctor.py)")
    return disabled


if __name__ == "__main__":
    d = apply(verbose=True)
    print("停用:", d or "(無, 環境正常)")
