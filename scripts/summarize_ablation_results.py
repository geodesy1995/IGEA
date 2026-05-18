import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, Optional

import pandas as pd


def parse_true_class_metrics(report_path: str) -> Dict[str, Optional[float]]:
    metrics = {
        "precision": None,
        "recall": None,
        "f1": None,
        "support": None,
    }
    if not os.path.exists(report_path):
        return metrics

    with open(report_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 5 and parts[0] in {"True", "1", "1.0"}:
                metrics["precision"] = float(parts[1])
                metrics["recall"] = float(parts[2])
                metrics["f1"] = float(parts[3])
                metrics["support"] = float(parts[4])
                break
    return metrics


def experiment_group(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IGEA ablation matrix outputs into a CSV table.")
    parser.add_argument("root", help="Root folder containing experiment output folders.")
    parser.add_argument(
        "--output",
        default="ablation_summary.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--raw-output",
        help="Optional CSV path for per-run rows before aggregation.",
    )
    args = parser.parse_args()

    rows = []
    metadata_paths = glob.glob(os.path.join(args.root, "**", "experiment metadata.json"), recursive=True)

    for metadata_path in metadata_paths:
        run_dir = os.path.dirname(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        report_metrics = parse_true_class_metrics(os.path.join(run_dir, "class_report.txt"))
        rows.append({
            "run_dir": run_dir,
            "kg_source": metadata.get("kg_source", ""),
            "experiment_name": metadata.get("experiment_name", ""),
            "experiment_group": experiment_group(metadata.get("experiment_name", "")),
            "spatial_variant": metadata.get("spatial_variant", ""),
            "spatial_features": ",".join(metadata.get("spatial_features", [])),
            "spatial_scaler": metadata.get("spatial_scaler", ""),
            "spatial_distance_transform": metadata.get("spatial_distance_transform", ""),
            "random_seed": metadata.get("random_seed", ""),
            "precision": report_metrics["precision"],
            "recall": report_metrics["recall"],
            "f1": report_metrics["f1"],
            "support": report_metrics["support"],
            "predicted_match_count": metadata.get("predicted_match_count", ""),
            "candidate_count": metadata.get("candidate_count", ""),
            "verifier_selected_count": metadata.get("verifier_selected_count", ""),
            "verifier_match_count": metadata.get("verifier_match_count", ""),
            "verifier_non_match_count": metadata.get("verifier_non_match_count", ""),
            "verifier_unsure_count": metadata.get("verifier_unsure_count", ""),
            "verifier_error_count": metadata.get("verifier_error_count", ""),
            "estimated_verifier_cost": metadata.get("estimated_verifier_cost", ""),
            "prediction_before_verifier_precision": metadata.get("prediction_before_verifier_precision", ""),
            "prediction_before_verifier_recall": metadata.get("prediction_before_verifier_recall", ""),
            "prediction_before_verifier_f1": metadata.get("prediction_before_verifier_f1", ""),
            "prediction_after_verifier_precision": metadata.get("prediction_after_verifier_precision", ""),
            "prediction_after_verifier_recall": metadata.get("prediction_after_verifier_recall", ""),
            "prediction_after_verifier_f1": metadata.get("prediction_after_verifier_f1", ""),
            "prediction_after_verifier_support": metadata.get("prediction_after_verifier_support", ""),
        })

    baseline_f1 = {}
    baseline_groups = {}
    baseline_prediction_f1 = {}
    baseline_prediction_groups = {}
    for row in rows:
        if row["spatial_variant"] == "original" and row["f1"] is not None:
            baseline_groups.setdefault(row["kg_source"], []).append(row["f1"])
        try:
            prediction_f1 = float(row["prediction_after_verifier_f1"])
            if row["spatial_variant"] == "original":
                baseline_prediction_groups.setdefault(row["kg_source"], []).append(prediction_f1)
        except (TypeError, ValueError):
            pass
    for kg_source, values in baseline_groups.items():
        baseline_f1[kg_source] = sum(values) / len(values)
    for kg_source, values in baseline_prediction_groups.items():
        baseline_prediction_f1[kg_source] = sum(values) / len(values)

    for row in rows:
        base = baseline_f1.get(row["kg_source"])
        row["delta_f1_vs_original"] = "" if base is None or row["f1"] is None else row["f1"] - base
        pred_base = baseline_prediction_f1.get(row["kg_source"])
        try:
            row["delta_prediction_f1_vs_original"] = (
                float(row["prediction_after_verifier_f1"]) - pred_base
                if pred_base is not None
                else ""
            )
        except (TypeError, ValueError):
            row["delta_prediction_f1_vs_original"] = ""
        try:
            row["delta_prediction_f1_vs_before_verifier"] = (
                float(row["prediction_after_verifier_f1"]) - float(row["prediction_before_verifier_f1"])
            )
        except (TypeError, ValueError):
            row["delta_prediction_f1_vs_before_verifier"] = ""

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
        "precision",
        "recall",
        "f1",
        "delta_f1_vs_original",
        "support",
        "predicted_match_count",
        "candidate_count",
        "verifier_selected_count",
        "verifier_match_count",
        "verifier_non_match_count",
        "verifier_unsure_count",
        "verifier_error_count",
        "estimated_verifier_cost",
        "prediction_before_verifier_precision",
        "prediction_before_verifier_recall",
        "prediction_before_verifier_f1",
        "prediction_after_verifier_precision",
        "prediction_after_verifier_recall",
        "prediction_after_verifier_f1",
        "delta_prediction_f1_vs_original",
        "delta_prediction_f1_vs_before_verifier",
        "prediction_after_verifier_support",
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
            "precision",
            "recall",
            "f1",
            "support",
            "predicted_match_count",
            "candidate_count",
            "verifier_selected_count",
            "verifier_match_count",
            "verifier_non_match_count",
            "verifier_unsure_count",
            "verifier_error_count",
            "estimated_verifier_cost",
            "prediction_before_verifier_precision",
            "prediction_before_verifier_recall",
            "prediction_before_verifier_f1",
            "prediction_after_verifier_precision",
            "prediction_after_verifier_recall",
            "prediction_after_verifier_f1",
            "prediction_after_verifier_support",
            "delta_prediction_f1_vs_before_verifier",
            "delta_prediction_f1_vs_original",
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
            ],
            dropna=False,
        )
        aggregate_rows = []
        for keys, group in grouped:
            kg_source, exp_group, spatial_variant, spatial_features, spatial_scaler, spatial_distance_transform = keys
            aggregate_rows.append({
                "kg_source": kg_source,
                "experiment_group": exp_group,
                "spatial_variant": spatial_variant,
                "spatial_features": spatial_features,
                "spatial_scaler": spatial_scaler,
                "spatial_distance_transform": spatial_distance_transform,
                "runs": int(len(group)),
                "seeds": ",".join(str(seed) for seed in sorted(group["random_seed"].dropna().astype(str).unique())),
                "precision_mean": group["precision"].mean(),
                "precision_std": group["precision"].std(ddof=0),
                "recall_mean": group["recall"].mean(),
                "recall_std": group["recall"].std(ddof=0),
                "f1_mean": group["f1"].mean(),
                "f1_std": group["f1"].std(ddof=0),
                "support_total": group["support"].sum(),
                "predicted_match_count_mean": group["predicted_match_count"].mean(),
                "candidate_count_mean": group["candidate_count"].mean(),
                "verifier_selected_count_total": group["verifier_selected_count"].sum(),
                "verifier_match_count_total": group["verifier_match_count"].sum(),
                "verifier_non_match_count_total": group["verifier_non_match_count"].sum(),
                "verifier_unsure_count_total": group["verifier_unsure_count"].sum(),
                "verifier_error_count_total": group["verifier_error_count"].sum(),
                "estimated_verifier_cost_total": group["estimated_verifier_cost"].sum(),
                "prediction_before_verifier_f1_mean": group["prediction_before_verifier_f1"].mean(),
                "prediction_after_verifier_f1_mean": group["prediction_after_verifier_f1"].mean(),
                "delta_prediction_f1_vs_original_mean": group["delta_prediction_f1_vs_original"].mean(),
                "delta_prediction_f1_vs_before_verifier_mean": group["delta_prediction_f1_vs_before_verifier"].mean(),
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
        "runs",
        "seeds",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "support_total",
        "predicted_match_count_mean",
        "candidate_count_mean",
        "verifier_selected_count_total",
        "verifier_match_count_total",
        "verifier_non_match_count_total",
        "verifier_unsure_count_total",
        "verifier_error_count_total",
        "estimated_verifier_cost_total",
        "prediction_before_verifier_f1_mean",
        "prediction_after_verifier_f1_mean",
        "delta_prediction_f1_vs_original_mean",
        "delta_prediction_f1_vs_before_verifier_mean",
        "delta_f1_vs_original_mean",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Wrote {len(aggregate_rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
