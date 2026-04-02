from __future__ import annotations

from pathlib import Path
from typing import Literal


DatasetName = Literal["Nextcloud", "WordPress"]
DatasetInput = Literal["Nextcloud", "WordPress", "Data", "Data_WP"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_MARKER = "GPT"


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
        raise ValueError(
            f"Unknown dataset={dataset!r}. Expected one of: {valid}"
        ) from exc


def experiment_aggregated_dir(dataset: DatasetInput | str = "Nextcloud") -> Path:
    canonical_dataset = normalize_dataset_name(dataset)
    return PROJECT_ROOT / "data" / canonical_dataset / "combine" / "ExperimentAggregated"


def is_ai_actor(actor: str, *, ai_marker: str = AI_MARKER) -> bool:
    return ai_marker in actor


def discover_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    root = experiment_aggregated_dir(dataset)
    if not root.exists():
        raise FileNotFoundError(f"Actor root not found: {root}")

    actors = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not actors:
        raise FileNotFoundError(f"No actor directories found under: {root}")
    return tuple(actors)


def discover_actor_groups(
    dataset: DatasetInput | str = "Nextcloud",
    *,
    ai_marker: str = AI_MARKER,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    actors = discover_actors(dataset)
    human_groups = tuple(actor for actor in actors if ai_marker not in actor)
    ai_groups = tuple(actor for actor in actors if ai_marker in actor)

    if not human_groups:
        raise ValueError(f"No human actor directories found for dataset={dataset!r}")
    if not ai_groups:
        raise ValueError(f"No AI actor directories found for dataset={dataset!r}")

    return human_groups, ai_groups


def analysis_actors(dataset: DatasetInput | str = "Nextcloud") -> tuple[str, ...]:
    human_groups, ai_groups = discover_actor_groups(dataset)
    return human_groups + ai_groups
