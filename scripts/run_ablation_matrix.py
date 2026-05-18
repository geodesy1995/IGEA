import argparse
import configparser
import os
import subprocess
import sys
import time
from typing import Dict, List

from experiment_config import SPATIAL_VARIANTS


DEFAULT_VARIANTS = [
    "original",
    "bearing",
    "offset",
    "topology",
    "all_spatial",
    "no_bearing",
    "no_offset",
    "no_topology",
    "all_spatial_dummy_gate",
]


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalized_data_folder(path: str) -> str:
    return path.replace("\\", "/").rstrip("/") + "/"


def configure_variant(base_config_path: str, variant: str, output_root: str, matrix_id: str) -> str:
    config = configparser.ConfigParser()
    config.read(base_config_path)

    if variant == "all_spatial_dummy_gate":
        spatial_variant = "all_spatial"
        verifier_enabled = True
    else:
        spatial_variant = variant
        verifier_enabled = False

    if spatial_variant not in SPATIAL_VARIANTS:
        raise ValueError(
            f"Unknown ablation variant '{variant}'. "
            f"Available variants: {', '.join(DEFAULT_VARIANTS)}"
        )

    kg_source = config.get("meta", "kg_source", fallback=os.path.splitext(os.path.basename(base_config_path))[0])
    experiment_name = f"{kg_source}_{variant}"
    run_dir = normalized_data_folder(os.path.join(output_root, matrix_id, experiment_name))
    config_dir = os.path.join(output_root, matrix_id, "configs")
    os.makedirs(config_dir, exist_ok=True)

    if not config.has_section("experiment"):
        config.add_section("experiment")
    config.set("experiment", "name", experiment_name)

    if not config.has_section("spatial context"):
        config.add_section("spatial context")
    config.set("spatial context", "variant", spatial_variant)
    config.set("spatial context", "features", ",".join(SPATIAL_VARIANTS[spatial_variant]))
    if not config.has_option("spatial context", "encoder_units"):
        config.set("spatial context", "encoder_units", "16,8")

    if not config.has_section("llm verifier"):
        config.add_section("llm verifier")
    config.set("llm verifier", "enabled", str(verifier_enabled))
    config.set("llm verifier", "provider", config.get("llm verifier", "provider", fallback="dummy"))
    config.set("llm verifier", "margin", config.get("llm verifier", "margin", fallback="0.10"))
    config.set("llm verifier", "cost_per_1k_calls", config.get("llm verifier", "cost_per_1k_calls", fallback="0"))
    config.set("llm verifier", "log_filename", config.get("llm verifier", "log_filename", fallback="llm verifier log.tsv"))

    config.set("meta", "data_folder", run_dir)

    output_config_path = os.path.join(config_dir, f"{experiment_name}.ini")
    with open(output_config_path, "w", encoding="utf-8") as file:
        config.write(file)
    return output_config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally execute IGEA ablation-matrix configs."
    )
    parser.add_argument("configs", nargs="+", help="Base config files, e.g. config/config.ini config/config_dbpedia.ini")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to generate.",
    )
    parser.add_argument(
        "--output-root",
        default="./data/ablation_matrix",
        help="Root folder for generated configs and run output folders.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run runExperiment.py for each generated config. Without this flag, only print commands.",
    )
    args = parser.parse_args()

    variants = split_csv(args.variants)
    matrix_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    generated_configs: Dict[str, str] = {}

    for base_config_path in args.configs:
        for variant in variants:
            generated_config = configure_variant(base_config_path, variant, args.output_root, matrix_id)
            generated_configs[f"{base_config_path}:{variant}"] = generated_config

    for label, generated_config in generated_configs.items():
        cmd = [sys.executable, "runExperiment.py", generated_config]
        print(f"{label}")
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
