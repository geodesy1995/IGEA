import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TRAIN_PAIRS_FILENAME = "train pairs.tsv"
UNMATCHED_PAIRS_FILENAME = "unmatched pairs.tsv"
SPATIAL_SCALER_FILENAME = "spatial scaler.sav"

AVAILABLE_SPATIAL_FEATURES = (
    "dist",
    "bearing_sin",
    "bearing_cos",
    "d_lat",
    "d_lon",
)

SPATIAL_VARIANTS: Dict[str, List[str]] = {
    "original": ["dist"],
    "distance": ["dist"],
    "distance_only": ["dist"],
    "bearing": ["dist", "bearing_sin", "bearing_cos"],
    "offset": ["dist", "d_lat", "d_lon"],
    "all_spatial": list(AVAILABLE_SPATIAL_FEATURES),
    "no_bearing": ["dist", "d_lat", "d_lon"],
    "no_offset": ["dist", "bearing_sin", "bearing_cos"],
}


@dataclass(frozen=True)
class VerifierSettings:
    enabled: bool
    provider: str
    margin: float
    cost_per_1k_calls: float
    log_filename: str
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 30.0
    max_output_tokens: int = 180
    max_calls: int = 0


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_spatial_variant(config) -> str:
    if config.has_section("spatial context") and config.has_option("spatial context", "variant"):
        variant = config.get("spatial context", "variant").strip()
        if variant:
            return variant
    return "original"


def get_spatial_features(config) -> List[str]:
    if config.has_section("spatial context") and config.has_option("spatial context", "features"):
        raw_features = config.get("spatial context", "features").strip()
        if raw_features:
            features = _split_csv(raw_features)
        else:
            features = SPATIAL_VARIANTS[get_spatial_variant(config)]
    else:
        variant = get_spatial_variant(config)
        if variant not in SPATIAL_VARIANTS:
            raise ValueError(
                f"Unknown spatial context variant '{variant}'. "
                f"Available variants: {', '.join(sorted(SPATIAL_VARIANTS))}"
            )
        features = SPATIAL_VARIANTS[variant]

    invalid = [feature for feature in features if feature not in AVAILABLE_SPATIAL_FEATURES]
    if invalid:
        raise ValueError(
            f"Invalid spatial feature(s): {', '.join(invalid)}. "
            f"Available features: {', '.join(AVAILABLE_SPATIAL_FEATURES)}"
        )
    if not features:
        raise ValueError("At least one spatial feature must be configured.")
    return list(features)


def get_spatial_encoder_units(config) -> List[int]:
    if not config.has_section("spatial context") or not config.has_option("spatial context", "encoder_units"):
        return [16, 8]
    raw_units = config.get("spatial context", "encoder_units").strip()
    if not raw_units:
        return []
    units = [int(unit) for unit in _split_csv(raw_units)]
    if any(unit <= 0 for unit in units):
        raise ValueError("spatial context encoder_units must be positive integers.")
    return units


def build_spatial_matrix(data: pd.DataFrame, features: Iterable[str]) -> np.ndarray:
    features = list(features)
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError(
            f"Missing configured spatial feature column(s): {', '.join(missing)}. "
            "Regenerate candidate pairs or select a compatible spatial context variant."
        )

    frame = data.loc[:, features].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "dist" in frame.columns:
        frame.loc[:, "dist"] = np.log1p(np.clip(frame["dist"], 0.0, None))
    values = frame.to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = 0.0
    return values


def fit_spatial_scaler(matrix: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(matrix)
    return scaler


def transform_spatial_matrix(matrix: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    values = scaler.transform(matrix).astype(np.float32)
    values[~np.isfinite(values)] = 0.0
    return values


def get_verifier_settings(config) -> VerifierSettings:
    if not config.has_section("llm verifier"):
        return VerifierSettings(
            enabled=False,
            provider="dummy",
            margin=0.10,
            cost_per_1k_calls=0.0,
            log_filename="llm verifier log.tsv",
        )

    return VerifierSettings(
        enabled=config.getboolean("llm verifier", "enabled", fallback=False),
        provider=config.get("llm verifier", "provider", fallback="dummy").strip() or "dummy",
        margin=config.getfloat("llm verifier", "margin", fallback=0.10),
        cost_per_1k_calls=config.getfloat("llm verifier", "cost_per_1k_calls", fallback=0.0),
        log_filename=config.get("llm verifier", "log_filename", fallback="llm verifier log.tsv").strip()
        or "llm verifier log.tsv",
        model=config.get("llm verifier", "model", fallback="gpt-4o-mini").strip() or "gpt-4o-mini",
        timeout_seconds=config.getfloat("llm verifier", "timeout_seconds", fallback=30.0),
        max_output_tokens=config.getint("llm verifier", "max_output_tokens", fallback=180),
        max_calls=config.getint("llm verifier", "max_calls", fallback=0),
    )


def get_experiment_name(config) -> str:
    if config.has_section("experiment") and config.has_option("experiment", "name"):
        name = config.get("experiment", "name").strip()
        if name:
            return name
    return get_spatial_variant(config)


def write_experiment_metadata(
    data_dir: str,
    config,
    spatial_features: List[str],
    verifier_settings: Optional[VerifierSettings] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    metadata_path = os.path.join(data_dir, "experiment metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    metadata = {
        **metadata,
        "experiment_name": get_experiment_name(config),
        "spatial_variant": get_spatial_variant(config),
        "spatial_features": spatial_features,
    }
    if config.has_section("meta") and config.has_option("meta", "kg_source"):
        metadata["kg_source"] = config.get("meta", "kg_source")
    if verifier_settings is not None:
        metadata["llm_verifier"] = {
            "enabled": verifier_settings.enabled,
            "provider": verifier_settings.provider,
            "margin": verifier_settings.margin,
            "cost_per_1k_calls": verifier_settings.cost_per_1k_calls,
            "log_filename": verifier_settings.log_filename,
            "model": verifier_settings.model,
            "timeout_seconds": verifier_settings.timeout_seconds,
            "max_output_tokens": verifier_settings.max_output_tokens,
            "max_calls": verifier_settings.max_calls,
        }
    if extra:
        metadata.update(extra)

    os.makedirs(data_dir, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
