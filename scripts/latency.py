"""Measure serving latency and memory from a real client.

Numbers in the README's cost section come from here rather than from memory.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import psutil
from fastapi.testclient import TestClient

from src.api.main import app

PAYLOAD = {
    "loan_amnt": 10000,
    "term": "36 months",
    "int_rate": 13.5,
    "installment": 300.0,
    "grade": "C",
    "sub_grade": "C3",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 65000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "dti": 18.0,
    "delinq_2yrs": 0,
    "fico_range_low": 680,
    "fico_range_high": 684,
    "inq_last_6mths": 1,
    "open_acc": 10,
    "revol_bal": 15000.0,
    "revol_util": 45.0,
    "total_acc": 20,
}
REQUESTS = 50


def main() -> None:
    process = psutil.Process()

    started = time.perf_counter()
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/predict", json=PAYLOAD)  # load the model
        load_seconds = time.perf_counter() - started

        timings = []
        for _ in range(REQUESTS):
            began = time.perf_counter()
            response = client.post("/predict", json=PAYLOAD)
            timings.append((time.perf_counter() - began) * 1000)
            if response.status_code != 200:
                raise SystemExit(f"/predict returned {response.status_code}")

    rss = process.memory_info().rss / 1048576
    report = {
        "model_load_seconds": round(load_seconds, 3),
        "rss_mb_after_load": round(rss, 1),
        "requests": REQUESTS,
        "p50_ms": round(statistics.median(timings), 2),
        "p95_ms": round(sorted(timings)[int(REQUESTS * 0.95) - 1], 2),
        "max_ms": round(max(timings), 2),
    }
    Path("docs/latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved docs/latency.json: {report}")


if __name__ == "__main__":
    main()
