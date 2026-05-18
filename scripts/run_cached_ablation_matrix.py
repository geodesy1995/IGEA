import argparse
import configparser
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from experiment_config import SPATIAL_VARIANTS
from run_ablation_matrix import DEFAULT_VARIANTS, split_csv


COMMON_FILES = [
    "osm rbf.tsv",
    "nca dataset.tsv",
    "qid_index.tsv",
    "predicted classes.tsv",
    "create view.sql",
    "wikidata classes.txt",
    "wikidata dump.parquet",
    "train pairs.tsv",
    "unmatched pairs.tsv",
    "generate_candidates_log.txt",
    "candidate_generation_audit.csv",
    "coverage_report.csv",
    "candidate_audit.csv",
]


def normalized_data_folder(path: str) -> str:
    return path.replace("\\", "/").rstrip("/") + "/"


def ensure_section(config: configparser.ConfigParser, section: str) -> None:
    if not config.has_section(section):
        config.add_section(section)


def write_config(config: configparser.ConfigParser, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        config.write(file)


def load_base_config(path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(path)
    kg_source = config.get("meta", "kg_source", fallback="").lower()
    if kg_source not in {"dbpedia", "wikidata"}:
        raise ValueError("run_cached_ablation_matrix.py supports DBpedia and Wikidata configs.")
    return config


def make_common_config(base: configparser.ConfigParser, data_folder: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict({section: dict(base.items(section)) for section in base.sections()})
    ensure_section(config, "experiment")
    kg_source = config.get("meta", "kg_source", fallback="kg").lower()
    config.set("experiment", "name", f"{kg_source}_common_cached")
    config.set("meta", "data_folder", normalized_data_folder(data_folder))
    config.set("meta", "num_iterations", "1")
    return config


def is_wikidata_direct(config: configparser.ConfigParser) -> bool:
    kg_source = config.get("meta", "kg_source", fallback="").lower()
    entity_source = config.get("wikidata scrape", "entity_source", fallback="country").lower()
    return kg_source == "wikidata" and entity_source == "osm_linked"


def make_variant_config(
    base: configparser.ConfigParser,
    variant: str,
    data_folder: str,
    seed: int,
) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict({section: dict(base.items(section)) for section in base.sections()})

    if variant == "all_spatial_dummy_gate":
        spatial_variant = "all_spatial"
        verifier_enabled = True
    else:
        spatial_variant = variant
        verifier_enabled = False

    if spatial_variant not in SPATIAL_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'")

    ensure_section(config, "experiment")
    kg_source = config.get("meta", "kg_source", fallback="kg").lower()
    config.set("experiment", "name", f"{kg_source}_{variant}_seed{seed}")

    ensure_section(config, "spatial context")
    config.set("spatial context", "variant", spatial_variant)
    config.set("spatial context", "features", ",".join(SPATIAL_VARIANTS[spatial_variant]))
    if not config.has_option("spatial context", "encoder_units"):
        config.set("spatial context", "encoder_units", "16,8")

    ensure_section(config, "llm verifier")
    config.set("llm verifier", "enabled", str(verifier_enabled))
    config.set("llm verifier", "provider", config.get("llm verifier", "provider", fallback="dummy"))
    config.set("llm verifier", "margin", config.get("llm verifier", "margin", fallback="0.10"))
    config.set("llm verifier", "cost_per_1k_calls", config.get("llm verifier", "cost_per_1k_calls", fallback="0"))
    config.set("llm verifier", "log_filename", config.get("llm verifier", "log_filename", fallback="llm verifier log.tsv"))

    config.set("meta", "data_folder", normalized_data_folder(data_folder))
    config.set("meta", "num_iterations", "1")
    config.set("meta", "random_seed", str(seed))
    return config


def run_step(script_path: str, *args: str) -> None:
    command = [sys.executable, script_path, *args]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def common_outputs_exist(common_it_dir: str) -> bool:
    required = [
        "nca dataset.tsv",
        "predicted classes.tsv",
        "wikidata classes.txt",
        "wikidata dump.parquet",
        "train pairs.tsv",
        "unmatched pairs.tsv",
    ]
    return all(os.path.exists(os.path.join(common_it_dir, filename)) for filename in required)


def resolve_common_it_dir(path: str) -> str:
    candidates = [
        path,
        os.path.join(path, "it_1"),
        os.path.join(path, "_common", "it_1"),
    ]
    for candidate in candidates:
        if common_outputs_exist(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find reusable common outputs. Pass an it_1 directory, a _common directory, "
        "or a previous matrix root containing _common/it_1."
    )


def copy_common_outputs(common_it_dir: str, variant_it_dir: str) -> None:
    os.makedirs(variant_it_dir, exist_ok=True)
    for filename in COMMON_FILES:
        src = os.path.join(common_it_dir, filename)
        dst = os.path.join(variant_it_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def read_candidate_generation_audit(common_it_dir: str) -> dict:
    path = os.path.join(common_it_dir, "candidate_generation_audit.csv")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[0] if rows else {}


def metric_float(metrics: dict, key: str) -> float:
    try:
        return float(metrics.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def labels_from_match_column(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.astype(np.float32).values
    return values.astype(str).str.lower().isin(["true", "1", "yes"]).astype(np.float32).values


def make_split_indices_for_audit(frame, labels, test_size, random_state, group_column, split_strategy, source_indices=None):
    if source_indices is None:
        source_indices = np.arange(len(frame))
    source_indices = np.asarray(source_indices)
    source_frame = frame.iloc[source_indices].reset_index(drop=True)
    source_labels = labels[source_indices]

    use_group_split = (
        split_strategy == "group"
        and group_column
        and group_column in source_frame.columns
        and source_frame[group_column].nunique(dropna=False) >= 3
    )

    if use_group_split:
        groups = source_frame[group_column].fillna("").astype(str).values
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_local, test_local = next(splitter.split(source_frame, source_labels, groups))
    else:
        label_counts = pd.Series(source_labels).value_counts()
        stratify = source_labels if len(label_counts) > 1 and label_counts.min() >= 2 else None
        train_local, test_local = train_test_split(
            np.arange(len(source_indices)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    return source_indices[train_local], source_indices[test_local], "group" if use_group_split else "row"


def positive_support_by_split(train_path: str, config: configparser.ConfigParser) -> dict:
    if not os.path.exists(train_path):
        return {}

    split_strategy = config.get("entity linking", "split_strategy", fallback="group").strip().lower()
    group_column = config.get("entity linking", "split_group_column", fallback="wkid").strip()
    header = pd.read_csv(train_path, sep="\t", nrows=0)
    if "match" not in header.columns:
        return {}
    usecols = ["match"]
    if group_column in header.columns:
        usecols.append(group_column)
    frame = pd.read_csv(train_path, sep="\t", usecols=usecols)
    labels = labels_from_match_column(frame["match"])
    seed_text = config.get("quality gates", "seeds", fallback=config.get("meta", "random_seed", fallback="42"))
    seeds = [int(seed) for seed in split_csv(seed_text)]

    support = {}
    for seed in seeds:
        train_full_idx, test_idx, split_used = make_split_indices_for_audit(
            frame,
            labels,
            0.20,
            seed,
            group_column,
            split_strategy,
        )
        train_idx, val_idx, val_split_used = make_split_indices_for_audit(
            frame,
            labels,
            0.10,
            seed,
            group_column,
            split_strategy,
            train_full_idx,
        )
        support[str(seed)] = {
            "split_strategy": split_used,
            "val_split_strategy": val_split_used,
            "train_true": int(np.sum(labels[train_idx] == 1.0)),
            "val_true": int(np.sum(labels[val_idx] == 1.0)),
            "test_true": int(np.sum(labels[test_idx] == 1.0)),
        }
    return support


def candidate_audit(common_it_dir: str, config: configparser.ConfigParser) -> tuple:
    train_path = os.path.join(common_it_dir, "train pairs.tsv")
    audit_path = os.path.join(common_it_dir, "candidate_audit.csv")
    min_train_true = config.getint("quality gates", "min_train_true", fallback=100)
    min_train_rows = config.getint("quality gates", "min_train_rows", fallback=2000)
    min_test_true_support = config.getint("quality gates", "min_test_true_support", fallback=20)
    require_nonzero_bbox = config.getboolean("quality gates", "require_nonzero_bbox", fallback=False)
    expected_dist_threshold = config.getint("candidate generation", "dist_threshold")
    expected_max_candidates = config.getint("candidate generation", "max_candidates")
    generation_audit = read_candidate_generation_audit(common_it_dir)

    metrics = {
        "train_rows": 0,
        "train_true": 0,
        "train_false": 0,
        "estimated_test_true_support": 0.0,
        "min_test_true_support_observed": 0,
        "positive_support_by_split": "{}",
        "candidate_generation_audit_present": bool(generation_audit),
        "gold_total": 0,
        "gold_within_2500m": 0,
        "gold_outside_2500m": 0,
        "gold_missing_after_candidate_generation": 0,
        "dropped_by_limit_count": 0,
        "bbox_nonzero_rows": 0,
        "passed": False,
        "failure_reasons": "",
    }
    metrics.update(generation_audit)

    if os.path.exists(train_path):
        for data in pd.read_csv(train_path, sep="\t", chunksize=50_000):
            metrics["train_rows"] += len(data)
            if "match" in data.columns:
                if data["match"].dtype == bool:
                    matches = data["match"]
                else:
                    matches = data["match"].astype(str).str.lower().isin(["true", "1", "yes"])
                metrics["train_true"] += int(matches.sum())
                metrics["train_false"] += int((~matches).sum())
            if "bbox_overlap" in data.columns:
                metrics["bbox_nonzero_rows"] += int((pd.to_numeric(data["bbox_overlap"], errors="coerce").fillna(0) != 0).sum())
        metrics["estimated_test_true_support"] = metrics["train_true"] * 0.20
        split_support = positive_support_by_split(train_path, config)
        if split_support:
            test_supports = [entry["test_true"] for entry in split_support.values()]
            metrics["min_test_true_support_observed"] = min(test_supports)
            metrics["positive_support_by_split"] = json.dumps(split_support, sort_keys=True)

    failures = []
    if not generation_audit:
        failures.append("candidate_generation_audit_missing")
    if generation_audit and metric_float(metrics, "dist_threshold") != float(expected_dist_threshold):
        failures.append("dist_threshold_mismatch")
    if generation_audit and metric_float(metrics, "max_candidates") != float(expected_max_candidates):
        failures.append("max_candidates_mismatch")
    if metrics["train_rows"] < min_train_rows:
        failures.append("train_rows")
    if metrics["train_true"] < min_train_true:
        failures.append("train_true")
    support_for_gate = metrics["min_test_true_support_observed"] or metrics["estimated_test_true_support"]
    if support_for_gate < min_test_true_support:
        failures.append("estimated_test_true_support")
    if metric_float(metrics, "gold_missing_after_candidate_generation") > 0:
        failures.append("gold_missing_after_candidate_generation")
    if require_nonzero_bbox and metrics["bbox_nonzero_rows"] == 0:
        failures.append("bbox_overlap")
    metrics["failure_reasons"] = ",".join(failures)
    metrics["passed"] = not failures

    with open(audit_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics["passed"], metrics


def run_candidate_generation_with_gate(common_it_dir: str, common_config: configparser.ConfigParser, common_config_path: str) -> None:
    run_step("./scripts/candidateGeneration.py", normalized_data_folder(common_it_dir), common_config_path)
    passed, metrics = candidate_audit(common_it_dir, common_config)
    if passed:
        print(f"Candidate quality gate passed: {metrics}", flush=True)
        return

    retry_dist = common_config.get("quality gates", "retry_dist_threshold", fallback="")
    retry_candidates = common_config.get("quality gates", "retry_max_candidates", fallback="")
    if retry_dist and retry_candidates:
        print(f"Candidate quality gate failed; retrying with dist_threshold={retry_dist}, max_candidates={retry_candidates}: {metrics}", flush=True)
        common_config.set("candidate generation", "dist_threshold", retry_dist)
        common_config.set("candidate generation", "max_candidates", retry_candidates)
        write_config(common_config, common_config_path)
        run_step("./scripts/candidateGeneration.py", normalized_data_folder(common_it_dir), common_config_path)
        passed, metrics = candidate_audit(common_it_dir, common_config)

    if not passed:
        raise RuntimeError(f"Candidate quality gate failed: {metrics}")
    print(f"Candidate quality gate passed after retry: {metrics}", flush=True)


def run_common_pipeline(common_it_dir: str, common_config: configparser.ConfigParser, common_config_path: str) -> None:
    os.makedirs(common_it_dir, exist_ok=True)
    kg_source = common_config.get("meta", "kg_source", fallback="dbpedia").lower()
    run_step("./scripts/prepareSchema.py", common_config_path)
    if is_wikidata_direct(common_config):
        run_step("./scripts/osm2rdf.py", normalized_data_folder(common_it_dir), common_config_path)
        run_step("./scripts/scrapeWikiData.py", normalized_data_folder(common_it_dir), common_config_path)
        run_step("./scripts/createDirectView.py", common_config_path)
        run_candidate_generation_with_gate(common_it_dir, common_config, common_config_path)
        return

    run_step("./scripts/osm2rdf.py", normalized_data_folder(common_it_dir), common_config_path)
    if kg_source == "wikidata":
        run_step("./scripts/readRDFWikidata.py", normalized_data_folder(common_it_dir), common_config_path)
    else:
        run_step("./scripts/readRDFDBpedia.py", normalized_data_folder(common_it_dir), common_config_path)
    run_step("./scripts/schemaMatch.py", normalized_data_folder(common_it_dir), common_config_path)
    if kg_source == "wikidata":
        run_step("./scripts/reformClasses.py", normalized_data_folder(common_it_dir), common_config_path)
        run_step("./scripts/scrapeWikiData.py", normalized_data_folder(common_it_dir), common_config_path)
    else:
        run_step("./scripts/reformClassesDBP.py", normalized_data_folder(common_it_dir), common_config_path)
        run_step("./scripts/scrapeDBPedia.py", normalized_data_folder(common_it_dir), common_config_path)
    run_candidate_generation_with_gate(common_it_dir, common_config, common_config_path)


def run_variant(variant_it_dir: str, variant_config_path: str, use_attention: bool) -> None:
    run_step("./scripts/prepareSchema.py", variant_config_path)
    if use_attention:
        run_step("./scripts/entityLinkingAttention.py", normalized_data_folder(variant_it_dir), variant_config_path)
    else:
        run_step("./scripts/computeFTEmbeddings.py", normalized_data_folder(variant_it_dir), variant_config_path)
        run_step("./scripts/entityLinking.py", normalized_data_folder(variant_it_dir), variant_config_path)
    run_step("./scripts/predictUnmatched.py", normalized_data_folder(variant_it_dir), variant_config_path, "1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run KG ablations while reusing common NCA/KG/candidate artifacts."
    )
    parser.add_argument("config", help="Base DBpedia or Wikidata config.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to run.",
    )
    parser.add_argument(
        "--output-root",
        default="./data/ablation_dbpedia_cached",
        help="Root folder for generated configs and outputs.",
    )
    parser.add_argument(
        "--reuse-common",
        action="store_true",
        help="Skip common pipeline if common outputs already exist.",
    )
    parser.add_argument(
        "--reuse-common-dir",
        help="Copy common outputs from an existing it_1, _common, or previous matrix directory.",
    )
    parser.add_argument(
        "--seeds",
        help="Comma-separated random seeds. Defaults to quality gates seeds or 42.",
    )
    args = parser.parse_args()

    base = load_base_config(args.config)
    variants: List[str] = split_csv(args.variants)
    seed_text = args.seeds or base.get("quality gates", "seeds", fallback="42")
    seeds = [int(seed) for seed in split_csv(seed_text)]
    matrix_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    matrix_root = os.path.join(args.output_root, matrix_id)
    config_dir = os.path.join(matrix_root, "configs")

    common_dir = os.path.join(matrix_root, "_common")
    common_it_dir = os.path.join(common_dir, "it_1")
    common_config = make_common_config(base, common_dir)
    kg_source = base.get("meta", "kg_source", fallback="kg").lower()
    common_config_path = os.path.join(config_dir, f"{kg_source}_common_cached.ini")
    write_config(common_config, common_config_path)

    if args.reuse_common_dir:
        source_common_it_dir = resolve_common_it_dir(args.reuse_common_dir)
        copy_common_outputs(source_common_it_dir, common_it_dir)
        print(f"Reusing common outputs from {source_common_it_dir}", flush=True)
        passed, metrics = candidate_audit(common_it_dir, common_config)
        if not passed:
            raise RuntimeError(f"Reusable common outputs failed candidate quality gate: {metrics}")
    elif args.reuse_common and common_outputs_exist(common_it_dir):
        print(f"Reusing common outputs from {common_it_dir}", flush=True)
        passed, metrics = candidate_audit(common_it_dir, common_config)
        if not passed:
            raise RuntimeError(f"Reusable common outputs failed candidate quality gate: {metrics}")
    else:
        run_common_pipeline(common_it_dir, common_config, common_config_path)

    use_attention = base.getboolean("entity linking", "attention", fallback=True)

    for variant in variants:
        for seed in seeds:
            variant_dir = os.path.join(matrix_root, f"{kg_source}_{variant}_seed{seed}")
            variant_it_dir = os.path.join(variant_dir, "it_1")
            variant_config = make_variant_config(base, variant, variant_dir, seed)
            variant_config_path = os.path.join(config_dir, f"{kg_source}_{variant}_seed{seed}.ini")
            write_config(variant_config, variant_config_path)
            copy_common_outputs(common_it_dir, variant_it_dir)
            run_variant(variant_it_dir, variant_config_path, use_attention)


if __name__ == "__main__":
    main()
