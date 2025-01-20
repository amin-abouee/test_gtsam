import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

data = []
with open(sys.argv[1], "r") as f:
    for line in f:
        values = [float(x) for x in line.strip().split()]
        data.append(values)

data = np.array(data)
data = pd.DataFrame(data, columns=["diff_time", "dx", "dy", "dz", "roll", "pitch", "yaw", "vel_x", "vel_y", "vel_z"])

# Create sample indices for x-axis since diff_time appears to be constant intervals
x_range = np.arange(len(data))

# Create hover templates for each type of data
time_template = "Value: %{y:.3f} s<br>Time: %{x:.3f} s"
pos_template = "Value: %{y:.3f} m<br>Time: %{x:.3f} s"
rot_template = "Value: %{y:.3f} rad<br>Time: %{x:.3f} s"
vel_template = "Value: %{y:.3f} m/s<br>Time: %{x:.3f} s"

fig = make_subplots(rows=4, cols=1, subplot_titles=("Time Difference", "Position", "Rotation", "Velocity"))

# Add time difference plot
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.diff_time,
        name="Time Diff",
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=time_template,
    ),
    row=1,
    col=1,
)

# Add position plot (x,y,z)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.dx,
        name="Pos X",
        line=dict(color="red"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=pos_template,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.dy,
        name="Pos Y",
        line=dict(color="green"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=pos_template,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.dz,
        name="Pos Z",
        line=dict(color="blue"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=pos_template,
    ),
    row=2,
    col=1,
)

# Add rotation plot (roll, pitch, yaw)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.roll,
        name="Rot Roll",
        line=dict(color="red"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=rot_template,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.pitch,
        name="Rot Pitch",
        line=dict(color="green"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=rot_template,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.yaw,
        name="Rot Yaw",
        line=dict(color="blue"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=rot_template,
    ),
    row=3,
    col=1,
)

# Add velocity plot
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.vel_x,
        name="Vel X",
        line=dict(color="red"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=vel_template,
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.vel_y,
        name="Vel Y",
        line=dict(color="green"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=vel_template,
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=data.vel_z,
        name="Vel Z",
        line=dict(color="blue"),
        mode="lines+markers",
        marker=dict(size=3),
        hovertemplate=vel_template,
    ),
    row=4,
    col=1,
)

fig.update_layout(title_text="Motion Data Visualization", autosize=True, hoverlabel=dict(align="left"))

# Link all subplots together
fig.update_xaxes(matches="x")  # This links all x-axes together
fig.update_yaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Position (m)", row=2, col=1)
fig.update_yaxes(title_text="Rotation (rad)", row=3, col=1)
fig.update_yaxes(title_text="Velocity (m/s)", row=4, col=1)


plot(fig, filename="states.html", auto_open=True)
