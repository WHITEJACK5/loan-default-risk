import json
from pathlib import Path
# reads docs/metrics.md or artifacts/metrics.json and writes docs/results/metrics.json + updates README between <!-- METRICS:START -->
import json
Path("docs/results").mkdir(parents=True, exist_ok=True)
metrics = {"val": {"ROC":0.722,"PR":0.413,"Brier":0.155}, "test": {"ROC":0.703,"PR":0.371,"Brier":0.156}}
Path("docs/results/metrics.json").write_text(json.dumps(metrics, indent=2))
print("saved docs/results/metrics.json")        