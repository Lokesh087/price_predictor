from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = REPORT_DIR / "model_results.csv"
TXT_FILE = REPORT_DIR / "metrics.txt"


def save_metrics(metrics):

    df = pd.DataFrame([metrics])

    try:

        if CSV_FILE.exists() and CSV_FILE.stat().st_size > 0:

            old = pd.read_csv(CSV_FILE)

            df = pd.concat([old, df], ignore_index=True)

    except EmptyDataError:
        pass

    df.to_csv(CSV_FILE, index=False)

    with open(TXT_FILE, "a") as file:

        file.write("=" * 50 + "\n")

        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

        file.write("\n")

    print("✅ Metrics Saved")
# from pathlib import Path
# import pandas as pd

# BASE_DIR = Path(__file__).resolve().parent.parent
# REPORT_DIR = BASE_DIR / "reports"

# REPORT_DIR.mkdir(parents=True, exist_ok=True)

# CSV_FILE = REPORT_DIR / "model_results.csv"
# TXT_FILE = REPORT_DIR / "metrics.txt"


# def save_metrics(metrics):

#     df = pd.DataFrame([metrics])

#     if CSV_FILE.exists():
#         old = pd.read_csv(CSV_FILE)
#         df = pd.concat([old, df], ignore_index=True)

#     df.to_csv(CSV_FILE, index=False)

#     with open(TXT_FILE, "a") as file:
#         file.write("=" * 50 + "\n")
#         for key, value in metrics.items():
#             file.write(f"{key}: {value}\n")
#         file.write("\n")