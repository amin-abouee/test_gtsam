import sys

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot


def load_scale_data(file_path):
    """Load scale data from text file"""
    data = np.loadtxt(file_path)
    return {
        "vo_norm": data[:, 0],
        "imu_norm": data[:, 1],
        "weighted_scale": data[:, 2],
        "log_weighted_scale": data[:, 3],
        "index": np.arange(len(data)),
    }


def create_scale_plot(data):
    """Create interactive Plotly visualization"""
    fig = go.Figure()

    # Add traces for each scale type
    fig.add_trace(
        go.Scatter(x=data["index"], y=data["vo_norm"], mode="lines", name="VO Norm", line=dict(color="royalblue"))
    )

    fig.add_trace(
        go.Scatter(x=data["index"], y=data["imu_norm"], mode="lines", name="IMU Norm", line=dict(color="firebrick"))
    )

    fig.add_trace(
        go.Scatter(
            x=data["index"],
            y=data["weighted_scale"],
            mode="lines",
            name="Weighted Scale",
            line=dict(color="forestgreen"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["index"],
            y=data["log_weighted_scale"],
            mode="lines",
            name="Log Weighted Scale",
            line=dict(color="darkorange"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Scale Visualization Over Time",
        xaxis_title="Index",
        yaxis_title="Scale Value",
        hovermode="x unified",
        legend_title="Scale Types",
    )

    plot(fig, filename="./scale.html", auto_open=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scale_visualization.py <data_file>")
        return

    file_path = sys.argv[1]
    data = load_scale_data(file_path)
    create_scale_plot(data)


if __name__ == "__main__":
    main()
