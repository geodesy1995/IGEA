import argparse
import configparser
import os
import shutil
import subprocess
import sys
import time
from typing import List

from experiment_config import SPATIAL_VARIANTS
from run_ablation_matrix import split_csv
from run_cached_ablation_matrix import normalized_data_folder, write_config


def margin_label(margin: float) -> str:
    return f"margin{int(round(margin * 1000)):03d}"


def all_spatial_it_dir(matrix_root: str, seed: int) -> str:
    return os.path.join(matrix_root, f"dbpedia_all_spatial_seed{seed}", "it_1")


def required_followup_inputs_exist(it_dir: str) -> bool:
    required = [
        "unmatched pairs.tsv",
        "keras model",
        "osm tokenizer.sav",
        "wikidata tokenizer.sav",
        "spatial scaler.sav",
    ]
    return all(os.path.exists(os.path.join(it_dir, name)) for name in required)


def resolve_source_matrix_root(path: str, seeds: List[int]) -> str:
    if all(required_followup_inputs_exist(all_spatial_it_dir(path, seed)) for seed in seeds):
        return path

    candidates = []
    if os.path.isdir(path):
        for entry in os.scandir(path):
            if entry.is_dir():
                candidates.append(entry.path)
    candidates.sort(key=lambda item: os.path.getmtime(item), reverse=True)

    for candidate in candidates:
        if all(required_followup_inputs_exist(all_spatial_it_dir(candidate, seed)) for seed in seeds):
            return candidate

    raise FileNotFoundError(
        "Could not find completed all_spatial seed outputs. Pass the timestamped full-run matrix root, "
        "for example data/ablation_dbpedia_osm_linked_full_leakage_free/20260518-071747."
    )


def make_followup_config(
    base: configparser.ConfigParser,
    variant: str,
    data_folder: str,
    seed: int,
    margin: float,
    model: str,
    max_calls: int,
    cost_per_1k_calls: float,
) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict({section: dict(base.items(section)) for section in base.sections()})

    if not config.has_section("experiment"):
        config.add_section("experiment")
    config.set("experiment", "name", f"dbpedia_{variant}_seed{seed}")

    if not config.has_section("spatial context"):
        config.add_section("spatial context")
    config.set("spatial context", "variant", variant)
    config.set("spatial context", "features", ",".join(SPATIAL_VARIANTS["all_spatial"]))

    if not config.has_section("llm verifier"):
        config.add_section("llm verifier")
    config.set("llm verifier", "enabled", "True")
    config.set("llm verifier", "provider", "openai")
    config.set("llm verifier", "margin", str(margin))
    config.set("llm verifier", "model", model)
    config.set("llm verifier", "max_calls", str(max_calls))
    config.set("llm verifier", "cost_per_1k_calls", str(cost_per_1k_calls))
    config.set("llm verifier", "log_filename", "llm verifier log.tsv")

    if not config.has_section("entity linking"):
        config.add_section("entity linking")
    config.set("entity linking", "write_predictions_to_db", "False")

    if not config.has_section("meta"):
        config.add_section("meta")
    config.set("meta", "data_folder", normalized_data_folder(data_folder))
    config.set("meta", "num_iterations", "1")
    config.set("meta", "random_seed", str(seed))

    return config


def run_predict_unmatched(variant_it_dir: str, config_path: str) -> None:
    command = [
        sys.executable,
        "./scripts/predictUnmatched.py",
        normalized_data_folder(variant_it_dir),
        config_path,
        "1",
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenAI real LLM verifier follow-up on completed all_spatial DBpedia runs."
    )
    parser.add_argument("config", help="Base DBpedia config.")
    parser.add_argument(
        "--source-matrix",
        required=True,
        help="Completed full-run timestamp root or its parent output root.",
    )
    parser.add_argument(
        "--output-root",
        default="./data/llm_verifier_followup_openai",
        help="Root folder for OpenAI verifier follow-up outputs.",
    )
    parser.add_argument("--margins", default="0.03,0.05,0.10", help="Comma-separated low-margin gate values.")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seeds to reuse.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for structured verifier outputs.")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Maximum verifier calls per follow-up run. Use 5 for smoke; 0 means unlimited.",
    )
    parser.add_argument(
        "--cost-per-1k-calls",
        type=float,
        default=0.0,
        help="Cost proxy per 1,000 verifier calls for reporting.",
    )
    args = parser.parse_args()

    base = configparser.ConfigParser()
    base.read(args.config)
    seeds = [int(seed) for seed in split_csv(args.seeds)]
    margins = [float(margin) for margin in split_csv(args.margins)]
    source_matrix_root = resolve_source_matrix_root(args.source_matrix, seeds)
    matrix_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    matrix_root = os.path.join(args.output_root, matrix_id)
    config_dir = os.path.join(matrix_root, "configs")

    print(f"Reusing all_spatial artifacts from {source_matrix_root}", flush=True)
    print(f"Writing OpenAI verifier follow-up to {matrix_root}", flush=True)

    for margin in margins:
        label = margin_label(margin)
        for seed in seeds:
            variant = f"all_spatial_real_llm_gate_{label}"
            source_it_dir = all_spatial_it_dir(source_matrix_root, seed)
            if not required_followup_inputs_exist(source_it_dir):
                raise FileNotFoundError(f"Missing reusable all_spatial artifacts for seed {seed}: {source_it_dir}")

            variant_dir = os.path.join(matrix_root, f"dbpedia_{variant}_seed{seed}")
            variant_it_dir = os.path.join(variant_dir, "it_1")
            shutil.copytree(source_it_dir, variant_it_dir, dirs_exist_ok=True)

            config = make_followup_config(
                base=base,
                variant=variant,
                data_folder=variant_dir,
                seed=seed,
                margin=margin,
                model=args.model,
                max_calls=args.max_calls,
                cost_per_1k_calls=args.cost_per_1k_calls,
            )
            config_path = os.path.join(config_dir, f"dbpedia_{variant}_seed{seed}.ini")
            write_config(config, config_path)
            run_predict_unmatched(variant_it_dir, config_path)


if __name__ == "__main__":
    main()
