import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional

import pandas as pd


def parse_class_report(report_path: str) -> Dict[str, Optional[float]]:
    result = {
        "true_precision": None,
        "true_recall": None,
        "true_f1": None,
        "true_support": None,
        "false_support": None,
    }
    if not os.path.exists(report_path):
        return result

    with open(report_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 5 and parts[0] in {"True", "1", "1.0"}:
                result["true_precision"] = float(parts[1])
                result["true_recall"] = float(parts[2])
                result["true_f1"] = float(parts[3])
                result["true_support"] = float(parts[4])
            elif len(parts) == 5 and parts[0] in {"False", "0", "0.0"}:
                result["false_support"] = float(parts[4])
    return result


def read_tsv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


def feature_stats(data: Optional[pd.DataFrame], features: List[str]) -> Dict[str, str]:
    if data is None:
        return {
            "constant_spatial_features": "",
            "nonzero_spatial_features": "",
        }

    constant = []
    nonzero = []
    for feature in features:
        if feature not in data.columns:
            constant.append(f"{feature}:missing")
            continue
        values = pd.to_numeric(data[feature], errors="coerce").fillna(0.0)
        if values.nunique(dropna=False) <= 1:
            constant.append(feature)
        if (values != 0).any():
            nonzero.append(feature)
    return {
        "constant_spatial_features": ",".join(constant),
        "nonzero_spatial_features": ",".join(nonzero),
    }


def boolean_match_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def audit_run(run_dir: str) -> Dict[str, object]:
    metadata_path = os.path.join(run_dir, "experiment metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    train = read_tsv(os.path.join(run_dir, "train pairs.tsv"))
    unmatched = read_tsv(os.path.join(run_dir, "unmatched pairs.tsv"))
    predictions = read_tsv(os.path.join(run_dir, "predicted entity matches.tsv"))
    report = parse_class_report(os.path.join(run_dir, "class_report.txt"))
    features = metadata.get("spatial_features", [])

    row = {
        "experiment_name": metadata.get("experiment_name", ""),
        "spatial_variant": metadata.get("spatial_variant", ""),
        "spatial_features": ",".join(features),
        "train_rows": "" if train is None else len(train),
        "train_true": "" if train is None or "match" not in train else int(boolean_match_series(train["match"]).sum()),
        "train_false": "" if train is None or "match" not in train else int((~boolean_match_series(train["match"])).sum()),
        "test_true_support": report["true_support"],
        "test_false_support": report["false_support"],
        "true_f1": report["true_f1"],
        "unmatched_rows": "" if unmatched is None else len(unmatched),
        "predicted_rows": "" if predictions is None else len(predictions),
        "predicted_match_count": metadata.get("predicted_match_count", ""),
        "candidate_count": metadata.get("candidate_count", ""),
        "verifier_selected_count": metadata.get("verifier_selected_count", ""),
        "spatial_scaler": metadata.get("spatial_scaler", ""),
        "spatial_distance_transform": metadata.get("spatial_distance_transform", ""),
        "run_dir": run_dir,
    }
    row.update(feature_stats(train, features))

    warnings = []
    if isinstance(row["train_rows"], int) and row["train_rows"] < 100:
        warnings.append("small_train_set")
    if report["true_support"] is not None and report["true_support"] < 10:
        warnings.append("small_true_test_support")
    if row["constant_spatial_features"]:
        warnings.append("constant_spatial_feature")
    row["warnings"] = ",".join(warnings)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IGEA ablation outputs for fragile or suspicious results.")
    parser.add_argument("root", help="Root directory containing ablation run outputs.")
    parser.add_argument("--output", default="ablation_diagnostics.csv", help="CSV output path.")
    args = parser.parse_args()

    metadata_paths = glob.glob(os.path.join(args.root, "**", "experiment metadata.json"), recursive=True)
    rows = [audit_run(os.path.dirname(path)) for path in metadata_paths]
    rows.sort(key=lambda row: (row["experiment_name"], row["run_dir"]))

    fieldnames = [
        "experiment_name",
        "spatial_variant",
        "spatial_features",
        "train_rows",
        "train_true",
        "train_false",
        "test_true_support",
        "test_false_support",
        "true_f1",
        "unmatched_rows",
        "predicted_rows",
        "predicted_match_count",
        "candidate_count",
        "constant_spatial_features",
        "nonzero_spatial_features",
        "spatial_scaler",
        "spatial_distance_transform",
        "verifier_selected_count",
        "warnings",
        "run_dir",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    warning_counts = {}
    for row in rows:
        for warning in row["warnings"].split(","):
            if warning:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
    if warning_counts:
        print("Warnings:")
        for warning, count in sorted(warning_counts.items()):
            print(f"- {warning}: {count}")


if __name__ == "__main__":
    main()
