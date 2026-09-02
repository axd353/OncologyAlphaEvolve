#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/nfs/home/adas23/projects/AlphaEvolve"
CONTROLLED_EVAL_DIR="$REPO_ROOT/Data/FunsearchEvaluatorDataAFRFocal/ControlledEvaluations"
OUTPUT_CSV="$CONTROLLED_EVAL_DIR/heldout_auc_roc_summary.csv"

IMBALANCE_LEVELS=(
  "Very_high_imbalance"
  "high_imbalance"
  "modest_imbalance"
)

cd "$REPO_ROOT"

for imbalance_level in "${IMBALANCE_LEVELS[@]}"; do
  config_path="$CONTROLLED_EVAL_DIR/$imbalance_level/my_configAFROnco.json"
  echo "Running evaluate_priofunction for $imbalance_level"
  PYTHONPATH="$PWD" python -m PostProcesingData.evaluate_priofunction "$config_path"
done

python - <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path

repo_root = Path("/nfs/home/adas23/projects/AlphaEvolve")
controlled_eval_dir = repo_root / "Data" / "FunsearchEvaluatorDataAFRFocal" / "ControlledEvaluations"
output_csv = controlled_eval_dir / "heldout_auc_roc_summary.csv"
imbalance_levels = [
    "Very_high_imbalance",
    "high_imbalance",
    "modest_imbalance",
]
target_ancestry = "AFRICAN_ANCESTRY"
main_method_column = "llm discovered"
baseline_columns = [
    "Mixture Learning",
    "Independent Learning Scheme",
    "TL-GDES",
    "TL-PR",
]
fieldnames = ["imbalance_level", main_method_column, *baseline_columns]


def ancestry_auc(ancestry_evaluations: list[dict[str, object]], fallback_auc: float) -> float:
    for evaluation in ancestry_evaluations:
        if str(evaluation.get("ancestry_group", "")).upper() == target_ancestry:
            return float(evaluation["auc_roc"])
    return fallback_auc


rows: list[dict[str, object]] = []
for imbalance_level in imbalance_levels:
    config_path = controlled_eval_dir / imbalance_level / "my_configAFROnco.json"
    config_payload = json.loads(config_path.read_text())
    prio_function_path = Path(config_payload["prio_function_path"]).expanduser().resolve()
    report_file_name = str(config_payload["report_file_name"])
    report_path = prio_function_path.parent / report_file_name
    report_payload = json.loads(report_path.read_text())
    results = report_payload["results"]
    row: dict[str, object] = {"imbalance_level": imbalance_level}
    row[main_method_column] = ancestry_auc(
        list(results.get("heldout_ancestry_evaluations", [])),
        float(results["heldout_auc_roc"]),
    )

    baselines_by_name = {
        str(baseline["name"]): baseline
        for baseline in list(results.get("baseline_evaluations", []))
    }
    for baseline_name in baseline_columns:
        baseline = baselines_by_name.get(baseline_name)
        if baseline is None:
            raise RuntimeError(
                f"Missing baseline {baseline_name!r} in report {report_path} for {imbalance_level}."
            )
        row[baseline_name] = ancestry_auc(
            list(baseline.get("heldout_ancestry_evaluations", [])),
            float(baseline["auc_roc"]),
        )
    rows.append(row)

with output_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {output_csv}")
PY
