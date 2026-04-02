import re
from pathlib import Path
from typing import Pattern, Sequence, List, Dict, Any, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt


from src.core.stats.data_catalog import get_log_path, analysis_actors



QUOTED_OR_BARE_VALUE_PATTERN = r'(?:\"[^"]*\"|\S+)'


def strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def is_ai_actor(label: str) -> bool:
    return "gpt" in label.lower()


def extract_rows_from_file(
    file_path: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = QUOTED_OR_BARE_VALUE_PATTERN,
    require_all_keys: bool = False,
    ignore_case: bool = False,
    keep_line: bool = True,
    strip_quotes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return row-aligned extracted data: one dict per matched log line.

    Each row contains:
      - "_lineno": 1-based line number in the file
      - "_line": original line (if keep_line=True)
      - extracted keys found in that line: {key: value, ...}

    If require_all_keys=True, only rows containing ALL keys are kept.
    """
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    flags = re.IGNORECASE if ignore_case else 0

    key_regexes = {
        k: re.compile(rf"\b{re.escape(k)}=(?P<val>{value_pattern})(?=\s|$)", flags=flags)
        for k in keys
    }

    rows: List[Dict[str, Any]] = []

    for lineno, line in enumerate(lines, start=1):
        if not prefix_regex.search(line):
            continue

        found: Dict[str, str] = {}
        for k, rx in key_regexes.items():
            m = rx.search(line)
            if m:
                val = m.group("val")
                if strip_quotes:
                    val = strip_outer_quotes(val)
                found[k] = val

        if require_all_keys and any(k not in found for k in keys):
            continue

        row: Dict[str, Any] = {"_lineno": lineno}
        if keep_line:
            row["_line"] = line
        row.update(found)

        rows.append(row)

    return rows


# ----------------------------------------------------------------------
# Co-occurrence helpers
# ----------------------------------------------------------------------

def conditional_value_counts(
    rows: List[Dict[str, Any]],
    *,
    given_key: str,
    given_value: str,
    target_key: str,
) -> Counter:
    """
    Count target_key values among rows where given_key == given_value.

    Example:
        given_key="a0", given_value="curl", target_key="a1"
    """
    return Counter(
        row[target_key]
        for row in rows
        if row.get(given_key) == given_value and target_key in row
    )


def pair_counts(
    rows: List[Dict[str, Any]],
    *,
    key1: str,
    key2: str,
) -> Counter:
    """
    Count co-occurring pairs (row[key1], row[key2]).
    """
    return Counter(
        (row[key1], row[key2])
        for row in rows
        if key1 in row and key2 in row
    )




# ----------------------------------------------------------------------
# Presence / missing-key-pattern helpers
# ----------------------------------------------------------------------

def row_presence_pattern(
    row: Dict[str, Any],
    keys: Sequence[str],
) -> Tuple[str, ...]:
    """
    Return tuple of keys present in this row, preserving key order.

    Example:
        keys=["a0","a1","a2"], row has only a0 and a2
        -> ("a0", "a2")
    """
    return tuple(k for k in keys if k in row)


def presence_pattern_counts(
    rows: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
) -> Counter:
    """
    Count how often each key-presence pattern appears.

    Example output keys:
        ("a0",)
        ("a0","a1")
        ("a0","a1","a2")
    """
    return Counter(row_presence_pattern(row, keys) for row in rows)


# ----------------------------------------------------------------------
# Distribution plotting helpers
# ----------------------------------------------------------------------


def _format_distribution_label(x: Any) -> str:
    if isinstance(x, tuple):
        return " | ".join(str(v) for v in x)
    return str(x)


def _counter_to_prob_dict(counter: Counter) -> Dict[Any, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def build_distribution_for_rows(
    rows: List[Dict[str, Any]],
    *,
    distribution_name: str,
    keys: Sequence[str],
    pair_key1: Optional[str] = None,
    pair_key2: Optional[str] = None,
    conditional_given_key: Optional[str] = None,
    conditional_given_value: Optional[str] = None,
    conditional_target_key: Optional[str] = None,
) -> Counter:
    """
    Build one distribution Counter from one actor's rows.
    """
    if distribution_name == "presence_pattern_distance":
        return presence_pattern_counts(rows, keys=keys)

    if distribution_name == "pair_distribution_distance":
        if pair_key1 is None or pair_key2 is None:
            raise ValueError("pair_key1 and pair_key2 are required")
        return pair_counts(rows, key1=pair_key1, key2=pair_key2)

    if distribution_name == "conditional_distribution_distance":
        if (
            conditional_given_key is None
            or conditional_given_value is None
            or conditional_target_key is None
        ):
            raise ValueError(
                "conditional_given_key, conditional_given_value and "
                "conditional_target_key are required"
            )
        return conditional_value_counts(
            rows,
            given_key=conditional_given_key,
            given_value=conditional_given_value,
            target_key=conditional_target_key,
        )

    raise ValueError(f"Unknown distribution_name: {distribution_name}")


def collect_actor_distributions(
    actor_files: Sequence[Tuple[str, str | Path]],
    *,
    distribution_name: str,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = QUOTED_OR_BARE_VALUE_PATTERN,
    require_all_keys: bool = False,
    ignore_case: bool = False,
    keep_line: bool = False,
    strip_quotes: bool = True,
    pair_key1: Optional[str] = None,
    pair_key2: Optional[str] = None,
    conditional_given_key: Optional[str] = None,
    conditional_given_value: Optional[str] = None,
    conditional_target_key: Optional[str] = None,
) -> Dict[str, Counter]:
    """
    Return {actor_label: Counter(...)} for selected actors.
    """
    result: Dict[str, Counter] = {}

    for label, file_path in actor_files:
        rows = extract_rows_from_file(
            file_path,
            prefix_regex=prefix_regex,
            keys=keys,
            value_pattern=value_pattern,
            require_all_keys=require_all_keys,
            ignore_case=ignore_case,
            keep_line=keep_line,
            strip_quotes=strip_quotes,
        )

        counter = build_distribution_for_rows(
            rows,
            distribution_name=distribution_name,
            keys=keys,
            pair_key1=pair_key1,
            pair_key2=pair_key2,
            conditional_given_key=conditional_given_key,
            conditional_given_value=conditional_given_value,
            conditional_target_key=conditional_target_key,
        )
        result[label] = counter

    return result


def select_actors(
    file_pairs: Sequence[Tuple[str, str | Path]],
    *,
    include_humans: Optional[int] = None,
    include_ais: Optional[int] = None,
    specific_actors: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str | Path]]:
    """
    Select actors either by exact names or by number of human/AI actors.
    """
    if specific_actors is not None:
        wanted = set(specific_actors)
        return [(label, path) for label, path in file_pairs if label in wanted]

    humans = [(label, path) for label, path in file_pairs if not is_ai_actor(label)]
    ais = [(label, path) for label, path in file_pairs if is_ai_actor(label)]

    selected: List[Tuple[str, str | Path]] = []

    if include_humans is not None:
        selected.extend(humans[:include_humans])

    if include_ais is not None:
        selected.extend(ais[:include_ais])

    return selected


def plot_actor_distributions(
    actor_distributions: Dict[str, Counter],
    *,
    top_k: int = 15,
    title: Optional[str] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot one grouped bar chart for multiple actors.

    X-axis:
        selected top-k distribution items

    For each item:
        one bar per actor
    """
    if not actor_distributions:
        print("No actor distributions to plot.")
        return

    global_counter = Counter()
    for counter in actor_distributions.values():
        global_counter.update(counter)

    vocab = [item for item, _ in global_counter.most_common(top_k)]
    if not vocab:
        print("No values to plot.")
        return

    labels = [_format_distribution_label(item) for item in vocab]
    actor_names = list(actor_distributions.keys())
    num_actors = len(actor_names)
    x = list(range(len(vocab)))

    plt.figure(figsize=(max(12, len(vocab) * 1.0), 6))

    group_width = 0.8
    bar_width = group_width / max(num_actors, 1)

    ylabel = "Probability" if normalize else "Count"

    for idx, actor in enumerate(actor_names):
        counter = actor_distributions[actor]

        if normalize:
            probs = _counter_to_prob_dict(counter)
            values = [probs.get(item, 0.0) for item in vocab]
        else:
            values = [counter.get(item, 0) for item in vocab]

        offsets = [
            i - group_width / 2 + bar_width / 2 + idx * bar_width
            for i in x
        ]
        plt.bar(offsets, values, width=bar_width, label=actor)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def get_mode_config(mode: str) -> Tuple[Pattern[str], List[str]]:
    if mode == "execve":
        return re.compile(r"^type=EXECVE\s+msg=audit\("), ["a0", "a1", "a2", "a3", "a4", "a5", "argc"]
    if mode == "path":
        return re.compile(r"^type=PATH\s+msg=audit\("), ["name", "nametype"]
    if mode == "syscall":
        return (
            re.compile(r"^type=SYSCALL\s+msg=audit\("),
            ["syscall", "success", "exit", "comm", "exe", "auid", "uid", "tty", "key"],
        )
    if mode == "sockaddr":
        return re.compile(r"^type=SOCKADDR\s+msg=audit\("), ["saddr"]
    raise ValueError(f"Unknown mode: {mode}")


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    use_wordpress = False
    dataset = "WordPress" if use_wordpress else "Nextcloud"

    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    ### Hyperparameter
    mode = "execve"
    pair_key1 = "a0"
    pair_key2 = "a1"
    conditional_given_key = "a0"
    conditional_given_value = "grep"
    conditional_target_key = "a1"
    sort_by = "conditional_distribution_distance"  # or presence_pattern_distance / pair_distribution_distance

    # multi-actor grouped plot
    actor_plot_top_k = 15

    # option A: specify exact actors
    specific_actors = ["Armin", "Benni", "Marvin", "Nico", "Hotti", "Torina", "GPT4.1", "GPT5", "GPT4.1_V2", "GPT4o"]

    # option B: take first n humans / ais if specific_actors=None
    include_humans = 2
    include_ais = 2

    # don't change this
    require_all_keys = False
    ignore_case = False
    keep_line = False
    strip_quotes = True
    value_pattern = QUOTED_OR_BARE_VALUE_PATTERN

    prefix, keys = get_mode_config(mode)

    selected_actor_files = select_actors(
        file_pairs,
        specific_actors=specific_actors,
        include_humans=include_humans,
        include_ais=include_ais,
    )

    actor_distributions = collect_actor_distributions(
        selected_actor_files,
        distribution_name=sort_by,
        prefix_regex=prefix,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
        keep_line=keep_line,
        strip_quotes=strip_quotes,
        pair_key1=pair_key1,
        pair_key2=pair_key2,
        conditional_given_key=conditional_given_key,
        conditional_given_value=conditional_given_value,
        conditional_target_key=conditional_target_key,
    )

    plot_actor_distributions(
        actor_distributions,
        top_k=actor_plot_top_k,
        title=f"{mode} {sort_by} per actor",
        normalize=True,
        save_path="results/actor_histograms.pdf",
    )


