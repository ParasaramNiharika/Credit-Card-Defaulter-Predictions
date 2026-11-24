
import csv
from datetime import datetime
from pathlib import Path


def log_prediction(customer, prediction, probability):

    project_root = Path(__file__).resolve().parents[1]
    log_folder = project_root / "logs"
    log_folder.mkdir(exist_ok=True)

    log_file = log_folder / "predictions.csv"

    record = customer.iloc[0].to_dict()
    record["timestamp"] = datetime.now().isoformat()
    record["prediction"] = int(prediction)
    record["default_probability"] = float(probability)

    file_exists = log_file.exists()

    with open(log_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=record.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)
