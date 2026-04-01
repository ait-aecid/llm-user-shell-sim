import matplotlib.pyplot as plt

from stats_tools.pairwise_viz import (
    build_symmetric_distance_matrix,
    plot_distance_heatmap,
    plot_mds_embedding,
)


def style_actor(label: str) -> dict:
    if label.startswith("H"):
        return {"color": "blue", "marker": "o"}
    return {"color": "red", "marker": "x"}


def run_example(pairwise_results, labels, title):
    matrix = build_symmetric_distance_matrix(
        labels,
        pairwise_results,
        extract_pair=lambda item: item,
    )

    print(f"\n{title}")
    plot_distance_heatmap(matrix, labels, title=f"{title} heatmap")
    _, stress = plot_mds_embedding(
        matrix,
        labels,
        title=f"{title} MDS",
        point_styler=style_actor,
    )
    print(f"Stress: {stress:.6f}")


labels = ["H1", "H2", "H3", "H4", "H5", "H6", "AI1", "AI2", "AI3", "AI4"]

good_pairs = [
    ("H1", "H2", 1.0), ("H1", "H3", 1.1), ("H1", "H4", 0.9), ("H1", "H5", 1.2), ("H1", "H6", 1.0),
    ("H1", "AI1", 5.2), ("H1", "AI2", 5.5), ("H1", "AI3", 5.8), ("H1", "AI4", 5.4),
    ("H2", "H3", 0.9), ("H2", "H4", 1.1), ("H2", "H5", 1.0), ("H2", "H6", 1.2),
    ("H2", "AI1", 5.1), ("H2", "AI2", 5.4), ("H2", "AI3", 5.7), ("H2", "AI4", 5.3),
    ("H3", "H4", 1.0), ("H3", "H5", 1.1), ("H3", "H6", 1.3),
    ("H3", "AI1", 5.3), ("H3", "AI2", 5.6), ("H3", "AI3", 5.9), ("H3", "AI4", 5.5),
    ("H4", "H5", 1.2), ("H4", "H6", 1.0),
    ("H4", "AI1", 4.9), ("H4", "AI2", 5.2), ("H4", "AI3", 5.4), ("H4", "AI4", 5.1),
    ("H5", "H6", 0.9),
    ("H5", "AI1", 5.4), ("H5", "AI2", 5.7), ("H5", "AI3", 6.0), ("H5", "AI4", 5.6),
    ("H6", "AI1", 5.0), ("H6", "AI2", 5.3), ("H6", "AI3", 5.6), ("H6", "AI4", 5.2),
    ("AI1", "AI2", 0.9), ("AI1", "AI3", 1.0), ("AI1", "AI4", 0.8),
    ("AI2", "AI3", 0.8), ("AI2", "AI4", 1.0),
    ("AI3", "AI4", 0.9),
]

bad_pairs = [
    ("H1", "H2", 2.4), ("H1", "H3", 2.7), ("H1", "H4", 2.3), ("H1", "H5", 2.5), ("H1", "H6", 2.6),
    ("H1", "AI1", 2.4), ("H1", "AI2", 2.6), ("H1", "AI3", 2.7), ("H1", "AI4", 2.5),
    ("H2", "H3", 2.5), ("H2", "H4", 2.6), ("H2", "H5", 2.3), ("H2", "H6", 2.7),
    ("H2", "AI1", 2.5), ("H2", "AI2", 2.4), ("H2", "AI3", 2.6), ("H2", "AI4", 2.3),
    ("H3", "H4", 2.4), ("H3", "H5", 2.6), ("H3", "H6", 2.5),
    ("H3", "AI1", 2.6), ("H3", "AI2", 2.5), ("H3", "AI3", 2.3), ("H3", "AI4", 2.4),
    ("H4", "H5", 2.7), ("H4", "H6", 2.5),
    ("H4", "AI1", 2.3), ("H4", "AI2", 2.6), ("H4", "AI3", 2.5), ("H4", "AI4", 2.4),
    ("H5", "H6", 2.4),
    ("H5", "AI1", 2.5), ("H5", "AI2", 2.4), ("H5", "AI3", 2.6), ("H5", "AI4", 2.3),
    ("H6", "AI1", 2.6), ("H6", "AI2", 2.5), ("H6", "AI3", 2.4), ("H6", "AI4", 2.7),
    ("AI1", "AI2", 2.4), ("AI1", "AI3", 2.7), ("AI1", "AI4", 2.5),
    ("AI2", "AI3", 2.5), ("AI2", "AI4", 2.6),
    ("AI3", "AI4", 2.4),
]

run_example(good_pairs, labels, "GOOD separation")
run_example(bad_pairs, labels, "BAD separation")

plt.close("all")
