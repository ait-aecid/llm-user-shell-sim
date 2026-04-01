from __future__ import annotations

import random
from itertools import product
from typing import List, Literal, Tuple, Optional

from core.loader import LoadConfig, resolve_human_ai_groups

DatasetName = Literal["Data", "Data_WP"]
SPLIT_SHUFFLE_SEED = 42


def make_human_ai_pairs(
    dataset: DatasetName = "Data",
    *,
    randomize_actor_labels: bool = False,
    assignment_idx: Optional[int] = None,
) -> List[List[str]]:
    cfg = LoadConfig(
        dataset=dataset,
        randomize_actor_labels=randomize_actor_labels,
        assignment_idx=assignment_idx,
    )
    human_groups, ai_groups = resolve_human_ai_groups(cfg)
    return [[h, a] for h, a in product(human_groups, ai_groups)]


def make_val_test_splits(
    dataset: DatasetName = "Data",
    *,
    randomize_actor_labels: bool = False,
    assignment_idx: Optional[int] = None,
) -> List[Tuple[List[str], List[str]]]:
    """
    Returns all (val_groups, test_groups) where:
      val_groups  = [human_v, ai_v]
      test_groups = [human_t, ai_t]
    and sets are disjoint => human_v != human_t AND ai_v != ai_t
    under the CURRENT label assignment.
    """
    cfg = LoadConfig(
        dataset=dataset,
        randomize_actor_labels=randomize_actor_labels,
        assignment_idx=assignment_idx,
    )
    human_groups, ai_groups = resolve_human_ai_groups(cfg)

    splits: List[Tuple[List[str], List[str]]] = []
    for hv, av in product(human_groups, ai_groups):
        for ht, at in product(human_groups, ai_groups):
            if hv == ht:
                continue
            if av == at:
                continue
            splits.append(([hv, av], [ht, at]))

    random.Random(SPLIT_SHUFFLE_SEED).shuffle(splits)
    return splits