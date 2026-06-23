"""從 Google Drive (公開資料夾) 批次下載除權息調整日線資料.

輸入: /tmp/drive_files.tsv  (title<TAB>fileId, 由 Drive 列舉產生)
輸出: data_adj/{code}.csv
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/drive_files.tsv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data_adj")


def download(title, fid):
    url = f"https://drive.google.com/uc?id={fid}&export=download"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            text = r.text
            if not text.startswith("date,"):
                raise ValueError("not csv")
            with open(os.path.join(OUT_DIR, title), "w") as f:
                f.write(text)
            return True
        except Exception:
            time.sleep(2 ** attempt)
    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = []
    with open(TSV) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0].endswith(".csv"):
                files.append(parts)
    print(f"{len(files)} files to download")

    ok, fail = 0, []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(download, t, i): t for t, i in files}
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail.append(futs[fut])
            if (ok + len(fail)) % 100 == 0:
                print(f"progress: {ok + len(fail)}/{len(files)}")
    print(f"done. ok={ok} fail={len(fail)} {fail[:10]}")


if __name__ == "__main__":
    main()
