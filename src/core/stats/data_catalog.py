from __future__ import annotations

from pathlib import Path
from src.core.shared.actor_catalog import (
    DatasetInput,
    analysis_actors as discover_analysis_actors,
    discover_actors,
    experiment_aggregated_dir,
)

LOG_FILE_NAMES = ("audit.log", "auth.log", "nextcloud.log", "syslog.log")


def known_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    return discover_actors(dataset)


def analysis_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    return discover_analysis_actors(dataset)


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
