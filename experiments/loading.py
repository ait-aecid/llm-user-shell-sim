from core.loader import load_examples, LoadConfig
from core.splits import make_splits
from core.reporting import print_split_stats
import numpy as np

def main():
    examples = load_examples(LoadConfig(
        log_files=("syslog.log",),   # ← ONLY audit
        prefix_with_log_type=False,
        preprocess_mode="aggressive",
        window_mode="none",         # windows are built from Drain3 cluster IDs
        window_size=50,
        window_stride=25,
        cid_prefix="CID",
    ))

    y = np.array([e.label for e in examples], dtype=object)
    groups = np.array([e.group for e in examples], dtype=object)

    split = make_splits(y, groups=groups, val_groups=["Armin", "GPT4.1"], test_groups=["GPT5", "GPT4.1_V2"])
    print_split_stats(examples, split)

    print(examples[0:50])

    print(y[0:50])

    print(groups[0:50])

    print(len(y))

if __name__ == "__main__":
    main()
