"""Run with python3 tests/test_eval_cli.py; stdlib only, no simulation/data required."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
environment = {**os.environ, "PRISM_PYTHON": sys.executable}
with tempfile.TemporaryDirectory(prefix="prism-eval-cli-") as directory:
    dataset = Path(directory)
    with (dataset / "clips.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "category"])
        writer.writeheader()
        writer.writerow({"clip_id": "ball", "category": "ball"})
    cases = [
        (["--help"], 0, "--data-dir"),
        (["--steps", "0"], 2, "--steps must be positive"),
        (["--demo", "--headless"], 2, "--demo requires a window"),
        (["--box-color", "1", "nan", "0"], 2, "--box-color values must be between"),
        (["--data-dir", str(dataset / "missing")], 2, "Dataset index not found"),
        (["--clip", "missing", "--data-dir", str(dataset)], 2, "--clip must name"),
        (["--clip", "ball", "--checkpoint", "box_23000.pt", "--data-dir", str(dataset)], 2, "Ball clips require"),
    ]
    (dataset / "video.json").write_text("{}")
    cases.append((["--record", str(dataset / "video.mp4")], 2, "Recording already exists"))
    for arguments, status, message in cases:
        result = subprocess.run(["bash", str(ROOT / "eval.sh"), *arguments],
                                cwd=directory, env=environment, capture_output=True, text=True)
        assert result.returncode == status, result.stdout + result.stderr
        assert message in result.stdout + result.stderr, result.stdout + result.stderr
print("PASS: eval CLI validation, sidecar preservation and launch outside repository")
