"""Minimal SWC reading and plotting for the teaching notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection


SWC_COLUMNS = ["node_id", "type", "x", "y", "z", "radius", "parent_id"]


def read_swc(path: str | Path) -> pd.DataFrame:
    """Read the seven standard SWC columns."""

    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=SWC_COLUMNS,
        usecols=range(7),
    )


def plot_swc_2d(nodes: pd.DataFrame, ax=None, color="#2b6cb0", title=None):
    """Draw parent-child edges in the x-y plane."""

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    by_id = nodes.set_index("node_id")
    segments = []
    for node in nodes.itertuples(index=False):
        if node.parent_id < 0 or node.parent_id not in by_id.index:
            continue
        parent = by_id.loc[node.parent_id]
        segments.append([(node.x, node.y), (parent.x, parent.y)])

    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or "SWC projection")
    return ax
