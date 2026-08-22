import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_intelligence.config import load_yaml
from churn_intelligence.data.loaders import load_raw_data
from churn_intelligence.data.validation import validate_dataset, write_validation_report

if __name__ == "__main__":
    cfg = load_yaml("data.yaml")
    frame = load_raw_data(ROOT / cfg["raw_path"])
    profile = validate_dataset(frame)
    write_validation_report(profile, ROOT / "results" / "reports")
    print(profile)
