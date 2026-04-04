import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

# HRNet (MMPose COCO-WholeBody) keypoint colors:
# left_ankle: [0, 255, 0], right_ankle: [255, 128, 0]
HRNET_LEFT_ANKLE_COLOR = "#00ff00"
HRNET_RIGHT_ANKLE_COLOR = "#ff8000"

def _as_index_array(v):
    if isinstance(v, tuple):
        v = v[0]
    return np.asarray(v, dtype=np.int64).ravel()

def plot_ankle_to_pelvis_distance(
    gait_data,
    gait_events,
    frame_indices,
    fps=60.0,
    output_path=None,
    show_plot=True,
):
    """
    plots a simple time series of the anterior-posterior location of the ankle to the central pelvis point, with markers for detected heel strikes and toe offs.
    """
    left_ankle_to_pelvis = gait_data["left_ankle_distance"]
    right_ankle_to_pelvis = gait_data["right_ankle_distance"]
    mid_pelvis_loc = gait_data["mid_pelvis_loc"]
    left_ankle_loc = gait_data["left_ankle"]
    right_ankle_loc = gait_data["right_ankle"]

    left_heel_strikes = _as_index_array(gait_events["left_heel_strikes"])
    left_toe_offs = _as_index_array(gait_events["left_toe_offs"])
    right_heel_strikes = _as_index_array(gait_events["right_heel_strikes"])
    right_toe_offs = _as_index_array(gait_events["right_toe_offs"])

    frame_indices = np.asarray(frame_indices, dtype=np.float64)
    time_seconds = frame_indices / float(fps)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(
        time_seconds,
        left_ankle_to_pelvis[:, 0],
        label="Left Ankle to Pelvis X",
        color=HRNET_LEFT_ANKLE_COLOR,
    )
    axes[0].plot(
        time_seconds,
        right_ankle_to_pelvis[:, 0],
        label="Right Ankle to Pelvis X",
        color=HRNET_RIGHT_ANKLE_COLOR,
    )

    axes[0].scatter(
        time_seconds[left_heel_strikes],
        left_ankle_to_pelvis[left_heel_strikes, 0],
        label="Left Heel Strikes",
        color=HRNET_LEFT_ANKLE_COLOR,
        marker="X",
        s=100,
    )
    axes[0].scatter(
        time_seconds[left_toe_offs],
        left_ankle_to_pelvis[left_toe_offs, 0],
        label="Left Toe Offs",
        color=HRNET_LEFT_ANKLE_COLOR,
        marker="o",
        s=100,
    )
    axes[0].scatter(
        time_seconds[right_heel_strikes],
        right_ankle_to_pelvis[right_heel_strikes, 0],
        label="Right Heel Strikes",
        color=HRNET_RIGHT_ANKLE_COLOR,
        marker="X",
        s=100,
    )
    axes[0].scatter(
        time_seconds[right_toe_offs],
        right_ankle_to_pelvis[right_toe_offs, 0],
        label="Right Toe Offs",
        color=HRNET_RIGHT_ANKLE_COLOR,
        marker="o",
        s=100,
    )

    axes[0].set_title('Ankle to Pelvis Distance Over Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Distance (pixels)')
    axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            frameon=True,
        )
    
    axes[0].grid()
    
    axes[1].plot(time_seconds, mid_pelvis_loc[:, 0], label='Mid Pelvis X', color='grey', alpha=0.5)
    axes[1].plot(
        time_seconds,
        left_ankle_loc[:, 0],
        label='Left Ankle X',
        color=HRNET_LEFT_ANKLE_COLOR,
    )
    axes[1].plot(
        time_seconds,
        right_ankle_loc[:, 0],
        label='Right Ankle X',
        color=HRNET_RIGHT_ANKLE_COLOR,
    )

    axes[1].set_title('Mid Pelvis and Ankle Locations Over Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Location (pixels)')
    axes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            frameon=True,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    axes[1].grid()

    # In remote/headless runs, interactive windows are not available.
    is_headless = os.environ.get("DISPLAY", "") == ""
    if output_path is None and is_headless:
        return fig

    if show_plot and not is_headless:
        plt.show()

    plt.close(fig)

def save_figure(fig, output_path):
    if output_path is not None:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        #check if plot already exists, if so, append a number to the filename
        if os.path.exists(output_path):
            base, ext = os.path.splitext(output_path)
            i = 1
            while os.path.exists(f"{base}_{i}{ext}"):
                i += 1
            output_path = f"{base}_{i}{ext}"
        fig.savefig(output_path)
        print(f"Saved gait events detection timeseries plot to: {output_path}")