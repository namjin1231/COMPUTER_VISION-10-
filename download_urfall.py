import shutil
import time
import urllib.request
from pathlib import Path

BASE = "https://fenix.ur.edu.pl/mkepski/ds/data/"
ROOT = Path("data/urfall")
TARGETS = [("fall", 30), ("adl", 40)]

def fetch(kind, n):
    name = f"{kind}-{n:02d}-cam0-rgb"
    dest_dir = ROOT / kind
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{name}.zip"
    if zip_path.exists() or (dest_dir / name).exists():
        print("건너뜀:", name)
        return
    part = zip_path.with_suffix(".part")
    req = urllib.request.Request(BASE + name + ".zip", headers={"User-Agent": "urfall-downloader"})
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(zip_path)
    print(f"받음: {name} ({zip_path.stat().st_size / 1e6:.0f}MB)")

if __name__ == "__main__":
    for kind, total in TARGETS:
        for n in range(1, total + 1):
            try:
                fetch(kind, n)
            except Exception as e:
                print(f"실패 {kind}-{n:02d}: {e!r}")
            time.sleep(1)