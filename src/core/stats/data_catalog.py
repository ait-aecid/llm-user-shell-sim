from __future__ import annotations

from pathlib import Path
from typing import Literal


DatasetName = Literal["Nextcloud", "WordPress"]
DatasetInput = Literal["Nextcloud", "WordPress", "Data", "Data_WP"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_FILE_NAMES = ("audit.log", "auth.log", "nextcloud.log", "syslog.log")
KNOWN_ACTORS_BY_DATASET: dict[DatasetName, tuple[str, ...]] = {
    "Nextcloud": ("Armin", "Benni", "GPT4.1", "GPT4.1_V2", "GPT4.1_V3", "GPT4o", "GPT5", "Hotti", "Marvin", "Nico", "Torina"),
    "WordPress": ("Armin", "GPT4.1", "GPT4.1_V2", "GPT4.1_V3", "GPT5", "Hotti", "Marvin", "Nico"),
}
ANALYSIS_ACTORS_BY_DATASET: dict[DatasetName, tuple[str, ...]] = {
    "Nextcloud": ("GPT4.1", "Hotti", "GPT4.1_V2", "GPT4o", "Armin", "Marvin", "Benni", "Nico", "Torina", "GPT5"),
    "WordPress": ("GPT4.1", "Hotti", "GPT4.1_V2", "GPT4.1_V3", "Armin", "Marvin", "Nico", "GPT5"),
}


def normalize_dataset_name(dataset: DatasetInput | str) -> DatasetName:
    aliases: dict[str, DatasetName] = {
        "Nextcloud": "Nextcloud",
        "WordPress": "WordPress",
        "Data": "Nextcloud",
        "Data_WP": "WordPress",
    }
    try:
        return aliases[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise KeyError(
            f"Unknown dataset: {dataset!r}. Expected one of: {valid}"
        ) from exc


def experiment_aggregated_dir(dataset: DatasetInput | str = "Nextcloud") -> Path:
    canonical_dataset = normalize_dataset_name(dataset)
    return PROJECT_ROOT / "data" / canonical_dataset / "combine" / "ExperimentAggregated"


def known_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    canonical_dataset = normalize_dataset_name(dataset)
    try:
        return KNOWN_ACTORS_BY_DATASET[canonical_dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(KNOWN_ACTORS_BY_DATASET))
        raise KeyError(f"Unknown dataset: {dataset!r}. Expected one of: {valid}") from exc


def analysis_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    canonical_dataset = normalize_dataset_name(dataset)
    try:
        return ANALYSIS_ACTORS_BY_DATASET[canonical_dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(ANALYSIS_ACTORS_BY_DATASET))
        raise KeyError(f"Unknown dataset: {dataset!r}. Expected one of: {valid}") from exc


def actor_dir(actor: str, dataset: DatasetInput | str = "Nextcloud") -> Path:
    return experiment_aggregated_dir(dataset) / actor


def get_actor_logs(actor: str, dataset: DatasetInput | str = "Nextcloud") -> dict[str, Path]:
    base = actor_dir(actor, dataset=dataset)
    return {
        "audit": base / "audit.log",
        "auth": base / "auth.log",
        "nextcloud": base / "nextcloud.log",
        "syslog": base / "syslog.log",
    }


def get_log_path(actor: str, log_name: str, dataset: DatasetInput | str = "Nextcloud") -> Path:
    key = log_name.removesuffix(".log")
    logs = get_actor_logs(actor, dataset=dataset)

    if key not in logs:
        valid = ", ".join(sorted(logs))
        raise KeyError(f"Unknown log name: {log_name!r}. Expected one of: {valid}")

    return logs[key]


def list_known_actor_logs(dataset: DatasetInput | str = "Nextcloud") -> dict[str, dict[str, Path]]:
    return {actor: get_actor_logs(actor, dataset=dataset) for actor in known_actors(dataset)}
