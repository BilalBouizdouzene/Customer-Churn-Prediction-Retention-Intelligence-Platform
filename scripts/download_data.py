from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "customer-churn-retention-intelligence/0.1"})
    with urlopen(request, timeout=60) as response:
        DEST.write_bytes(response.read())
    print(f"Downloaded {DEST} from {URL}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise
