# sea/tools/_lib/colours.py
"""
Dark-theme colour helpers for Matplotlib and Plotly charts.
Migrated from ADAPT2/tools/playground/colours.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

_BACKGROUND = "#2B0030"
_PALETTE = "viridis"
_N_COLORS = 4


def create_colored_plot():
    """Return (fig, ax, hex_colors) with a dark purple background."""
    colors = sns.color_palette(_PALETTE, _N_COLORS).as_hex()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(_BACKGROUND)
    fig.set_facecolor(_BACKGROUND)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    return fig, ax, colors


def create_colored_plotly():
    """Return (fig, hex_colors) with a dark purple Plotly layout."""
    colors = sns.color_palette(_PALETTE, _N_COLORS).as_hex()
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor=_BACKGROUND,
        paper_bgcolor=_BACKGROUND,
        font=dict(color="white"),
        xaxis=dict(showgrid=False, zeroline=False, color="white",
                   tickcolor="white", tickfont=dict(color="white")),
        yaxis=dict(showgrid=False, zeroline=False, color="white",
                   tickcolor="white", tickfont=dict(color="white")),
        title=dict(font=dict(color="white")),
    )
    fig.update_traces(marker=dict(color=colors[0]))
    return fig, colors
