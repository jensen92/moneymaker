"""環境健檢: 診斷 numpy / pandas / numexpr / bottleneck 是否相容, 並印出根治指令.

用途: 當 Telegram 掃描或回測噴出
    "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x ..."
這類錯誤時, 執行:
    python3 doctor.py
它會列出各套件版本、判斷哪個與你的 numpy 不相容, 並給出最適合的修復指令。

注意: numexpr / bottleneck 只是 pandas 的「選用加速器」, 不是必要相依。最簡單且零風險的
修法就是把這兩個壞掉的舊版本移除 (pandas 在它們缺席時功能完全正常)。
"""
import importlib.metadata as meta
import sys


def _ver(pkg):
    try:
        return meta.version(pkg)
    except meta.PackageNotFoundError:
        return None


def _major(v):
    try:
        return int(v.split(".")[0])
    except Exception:
        return None


def _try_import(name):
    """回傳 (ok, 訊息). ok=False 代表匯入時出錯 (常見為 numpy ABI 不合)."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # 把 numpy ABI 警告升級成錯誤以便偵測
            __import__(name)
        return True, "OK"
    except BaseException as e:                        # noqa: BLE001
        return False, f"{e.__class__.__name__}: {str(e).splitlines()[0][:80]}"


def main():
    print("=" * 60)
    print("moneymaker 環境健檢 (doctor.py)")
    print("=" * 60)
    print(f"Python      {sys.version.split()[0]}  ({sys.executable})")

    np_v = _ver("numpy")
    pd_v = _ver("pandas")
    ne_v = _ver("numexpr")
    bn_v = _ver("bottleneck")
    print(f"numpy       {np_v or '(未安裝)'}")
    print(f"pandas      {pd_v or '(未安裝)'}")
    print(f"numexpr     {ne_v or '(未安裝)'}")
    print(f"bottleneck  {bn_v or '(未安裝)'}")
    print("-" * 60)

    if not np_v or not pd_v:
        print("❌ 缺少 numpy 或 pandas, 請先: pip install numpy pandas requests")
        return

    np_major = _major(np_v)
    broken = []

    # 逐一試載入選用加速套件, 偵測 ABI 不合
    for name, v in (("numexpr", ne_v), ("bottleneck", bn_v)):
        if v is None:
            print(f"• {name:11} 未安裝 → pandas 會自動跳過 (無妨)")
            continue
        ok, msg = _try_import(name)
        if ok:
            print(f"• {name:11} {v}  匯入正常 ✅")
        else:
            broken.append((name, v))
            print(f"• {name:11} {v}  匯入失敗 ❌  ({msg})")

    print("-" * 60)
    if not broken:
        print("✅ 環境正常, 沒有偵測到不相容套件。若仍有問題請貼完整訊息。")
        return

    names = " ".join(n for n, _ in broken)
    print("❌ 偵測到與目前 numpy %s 不相容的舊版加速套件: %s" %
          (np_v, ", ".join(n for n, _ in broken)))
    print()
    print("這些只是 pandas 的『選用加速器』, 移除或升級皆可, pandas 功能不受影響。")
    print("請擇一執行 (建議由上往下試):")
    print()
    print("  【最簡單・零風險】移除壞掉的加速套件 (pandas 會自動以純 Python 後備):")
    print(f"      pip uninstall -y {names}")
    print()
    print("  【保留加速】升級到支援 numpy 2.x 的版本:")
    print(f"      pip install --upgrade 'numexpr>=2.10' 'bottleneck>=1.4'")
    print()
    if np_major and np_major >= 2:
        print("  【或】把 numpy 降回 1.x (其它套件若也只支援 1.x 時):")
        print("      pip install 'numpy<2'")
        print()
    print("anaconda 使用者若上述 pip 無效, 可改用 conda:")
    print(f"      conda install -c conda-forge {names}    # 升級到相容版")
    print()
    print("修好後, 機器人 (telegram_bot.py) 已內建 env_guard 會自動停用任何殘留的壞套件,")
    print("掃描不會再噴 NumPy 1.x 警告。")


if __name__ == "__main__":
    main()
