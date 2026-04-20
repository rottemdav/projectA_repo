import os
from datetime import datetime

def generate_output_paths(config):
    """
    Generates output paths for video and JSON files based on the global config.

    Args:
        config: The global configuration object.

    Returns:
        A dictionary containing paths for 'video', 'json', 'filtered_video', etc.
    """
    video_name = os.path.splitext(os.path.basename(config.INPUT_PATH))[0]
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S') # Or use config.DATE if it's static per run

    # Determine the frame range string for the output filename
    if config.END_FRAME is not None:
        out_range = f"{config.START_FRAME}_to_{config.END_FRAME}"
    elif config.MAX_FRAMES is not None:
        out_range = f"{config.START_FRAME}_to_{config.START_FRAME + config.MAX_FRAMES - 1}"
    else:
        out_range = f"{config.START_FRAME}_to_end"

    # A dictionary to hold all generated paths
    paths = {}

    # Use the format strings from the config to build the paths
    paths['video'] = os.path.join(
        config.OUTPUT_DIR,
        config.VIDEO_FILENAME_FORMAT.format(video_name=video_name, DATE=date_str, out_range=out_range)
    )
    paths['json'] = os.path.join(
        config.OUTPUT_DIR,
        config.JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=date_str, out_range=out_range)
    )
    paths['filtered_video'] = os.path.join(
        config.OUTPUT_DIR,
        config.FILTERED_VIDEO_FILENAME_FORMAT.format(video_name=video_name, DATE=date_str, out_range=out_range)
    )
    paths['filtered_json'] = os.path.join(
        config.OUTPUT_DIR,
        config.FILTERED_JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=date_str, out_range=out_range)
    )

    return paths