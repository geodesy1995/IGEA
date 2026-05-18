import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, Optional

import pandas as pd

from experiment_config import TEST_PREDICTIONS_FILENAME


def experiment_group(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name or "")


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def positive_metrics(frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    if frame.empty or "match" not in frame.columns or "prediction" not in frame.columns:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "support": 0,
            "rows": 0,
            "predicted_match_count": 0,
        }

    y_true = bool_series(frame["match"])
    y_pred = bool_series(frame["prediction"])
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    support = int(y_true.sum())
    predicted = int(y_pred.sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "rows": int(len(frame)),
        "predicted_match_count": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize positive-class metrics for near-field test predictions."
    )
    parser.add_argument("root", help="Root folder containing experiment output folders.")
    parser.add_argument("--max-distance", type=float, default=400.0, help="Near-field distance threshold in meters.")
    parser.add_argument("--output", default="near_field_summary_400m.csv", help="Aggregate CSV output path.")
    parser.add_argument("--raw-output", help="Optional per-run CSV output path.")
    args = parser.parse_args()

    rows = []
    metadata_paths = glob.glob(os.path.join(args.root, "**", "experiment metadata.json"), recursive=True)
    for metadata_path in metadata_paths:
        run_dir = os.path.dirname(metadata_path)
        predictions_path = os.path.join(run_dir, TEST_PREDICTIONS_FILENAME)
        if not os.path.exists(predictions_path):
            continue

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        predictions = pd.read_csv(predictions_path, sep="\t")
        if "dist" not in predictions.columns:
            filtered = predictions.iloc[0:0].copy()
        else:
            dist = pd.to_numeric(predictions["dist"], errors="coerce")
            filtered = predictions[dist <= args.max_distance].copy()

        metrics = positive_metrics(filtered)
        rows.append({
            "kg_source": metadata.get("kg_source", ""),
            "experiment_name": metadata.get("experiment_name", ""),
            "experiment_group": experiment_group(metadata.get("experiment_name", "")),
            "spatial_variant": metadata.get("spatial_variant", ""),
            "spatial_features": ",".join(metadata.get("spatial_features", [])),
            "spatial_scaler": metadata.get("spatial_scaler", ""),
            "spatial_distance_transform": metadata.get("spatial_distance_transform", ""),
            "random_seed": metadata.get("random_seed", ""),
            "near_field_max_distance_m": args.max_distance,
            "near_field_rows": metrics["rows"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "support": metrics["support"],
            "predicted_match_count": metrics["predicted_match_count"],
            "run_dir": run_dir,
        })

    baseline_f1 = {}
    baseline_groups = {}
    for row in rows:
        if row["spatial_variant"] == "original" and row["f1"] is not None:
            baseline_groups.setdefault(row["kg_source"], []).append(row["f1"])
    for kg_source, values in baseline_groups.items():
        baseline_f1[kg_source] = sum(values) / len(values)

    for row in rows:
        base = baseline_f1.get(row["kg_source"])
        row["delta_f1_vs_original"] = "" if base is None or row["f1"] is None else row["f1"] - base

    rows.sort(key=lambda row: (row["kg_source"], row["experiment_name"], row["run_dir"]))
    fieldnames = [
        "kg_source",
        "experiment_name",
        "experiment_group",
        "spatial_variant",
        "spatial_features",
        "spatial_scaler",
        "spatial_distance_transform",
        "random_seed",
        "near_field_max_distance_m",
        "near_field_rows",
        "precision",
        "recall",
        "f1",
        "delta_f1_vs_original",
        "support",
        "predicted_match_count",
        "run_dir",
    ]

    if args.raw_output:
        with open(args.raw_output, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if not rows:
        aggregate_rows = []
    else:
        frame = pd.DataFrame(rows)
        numeric_cols = [
            "near_field_rows",
            "precision",
            "recall",
            "f1",
            "support",
            "predicted_match_count",
            "delta_f1_vs_original",
        ]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        grouped = frame.groupby(
            [
                "kg_source",
                "experiment_group",
                "spatial_variant",
                "spatial_features",
                "spatial_scaler",
                "spatial_distance_transform",
                "near_field_max_distance_m",
            ],
            dropna=False,
        )
        aggregate_rows = []
        for keys, group in grouped:
            (
                kg_source,
                exp_group,
                spatial_variant,
                spatial_features,
                spatial_scaler,
                spatial_distance_transform,
                near_field_max_distance_m,
            ) = keys
            aggregate_rows.append({
                "kg_source": kg_source,
                "experiment_group": exp_group,
                "spatial_variant": spatial_variant,
                "spatial_features": spatial_features,
                "spatial_scaler": spatial_scaler,
                "spatial_distance_transform": spatial_distance_transform,
                "near_field_max_distance_m": near_field_max_distance_m,
                "runs": int(len(group)),
                "seeds": ",".join(str(seed) for seed in sorted(group["random_seed"].dropna().astype(str).unique())),
                "near_field_rows_total": group["near_field_rows"].sum(),
                "precision_mean": group["precision"].mean(),
                "precision_std": group["precision"].std(ddof=0),
                "recall_mean": group["recall"].mean(),
                "recall_std": group["recall"].std(ddof=0),
                "f1_mean": group["f1"].mean(),
                "f1_std": group["f1"].std(ddof=0),
                "support_total": group["support"].sum(),
                "predicted_match_count_mean": group["predicted_match_count"].mean(),
                "delta_f1_vs_original_mean": group["delta_f1_vs_original"].mean(),
            })
        aggregate_rows.sort(key=lambda row: (row["kg_source"], row["spatial_variant"]))

    aggregate_fieldnames = [
        "kg_source",
        "experiment_group",
        "spatial_variant",
        "spatial_features",
        "spatial_scaler",
        "spatial_distance_transform",
        "near_field_max_distance_m",
        "runs",
        "seeds",
        "near_field_rows_total",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "support_total",
        "predicted_match_count_mean",
        "delta_f1_vs_original_mean",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Wrote {len(aggregate_rows)} near-field aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
