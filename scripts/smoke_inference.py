"""Check the real bundle, optionally comparing every demo row with a live API."""
import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference_pipeline import load_inference_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", help="For example http://127.0.0.1:8000")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/production/inference_pipeline.pkl")
    args = parser.parse_args()
    pipeline = load_inference_pipeline(args.model)
    samples = pd.read_csv(ROOT / "src/display_streamlit/models_streamlit/demo_scenarios.csv", index_col=0)
    if args.api_url:
        with urlopen(args.api_url.rstrip("/") + "/health", timeout=30) as response:
            assert json.load(response)["status"] == "ok"
    predictions = []
    for row in json.loads(samples.to_json(orient="records")):
        settings = {
            "month": int(row["Month"]), "weekday": int(row["Weekday"]), "hour": int(row["Hour"]),
            "incident_group": row["IncidentGroup"], "special_service_type": row["SpecialServiceType"],
            "property_category": row["PropertyCategory"], "property_type": row["PropertyType"],
        }
        expected = float(pipeline.predict_scenario(row, settings)[0])
        assert np.isfinite(expected) and expected >= 0
        if args.api_url:
            request = Request(args.api_url.rstrip("/") + "/predict",
                              data=json.dumps({"template": row, **settings}).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=30) as response:
                actual = json.load(response)["attendance_time_seconds"]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)
        predictions.append(expected)
    print(f"Passed {len(predictions)} scenarios; predictions {min(predictions):.2f}–{max(predictions):.2f} seconds")
    if args.api_url:
        print("All HTTP predictions match the local inference bundle.")


if __name__ == "__main__":
    main()
