"""Small plotting helpers used by the main notebook."""

import matplotlib.pyplot as plt


REGION_COLORS = {"Cortex": "#2b6cb0", "Thalamus": "#dd6b20", "CP": "#319795"}


def plot_top_relevance(relevance, n=10, ax=None):
    """Horizontal bar chart of the most target-relevant features."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    top = relevance.head(n).sort_values()
    ax.barh(top.index, top.values, color="#4c78a8")
    ax.set_xlabel("mutual information")
    ax.set_title("Maximum-relevance ranking")
    return ax
