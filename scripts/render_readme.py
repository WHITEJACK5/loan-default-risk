import json
from pathlib import Path
metrics = json.loads(Path("artifacts/metrics.json").read_text(encoding="utf-8"))
block = f"| val | {metrics['val']['ROC']:.3f} | {metrics['val']['PR']:.3f} | {metrics['val']['Brier']:.3f} |\n| test | {metrics['test']['ROC']:.3f} | {metrics['test']['PR']:.3f} | {metrics['test']['Brier']:.3f} |"
readme = Path("README.md").read_text(encoding="utf-8")
start, end = "<!-- METRICS:START -->", "<!-- METRICS:END -->"
if start in readme:
    before, rest = readme.split(start)
    _, after = rest.split(end)
    readme = before + start + "\n" + block + "\n" + end + after
    Path("README.md").write_text(readme, encoding="utf-8")
    print("rendered README metrics")
