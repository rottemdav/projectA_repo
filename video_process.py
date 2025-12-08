import sys
import cv2
import os
import numpy as np

# 1. SETUP PATHS
# Use os.path.join to avoid slash errors
project_root = r"C:\Users\Rottem2\Desktop\Rottem\school\project_a\openpose"
python_openpose_path = os.path.join(project_root, 'build', 'python', 'openpose', 'Release')
x64_release_path = os.path.join(project_root, 'build', 'x64', 'Release')
bin_path = os.path.join(project_root, 'build', 'bin')

# 2. APPEND TO SYS.PATH (To find the python script)
sys.path.append(python_openpose_path)

# 3. LOAD DLLs (Crucial fix for Python 3.8+)
# Windows requires you to explicitly trust these directories for DLL loading
try:
    if os.path.exists(x64_release_path):
        os.add_dll_directory(x64_release_path)
    if os.path.exists(bin_path):
        os.add_dll_directory(bin_path)
    # Add legacy PATH support just in case
    os.environ['PATH'] = os.environ['PATH'] + ';' + x64_release_path + ';' + bin_path
except AttributeError:
    # Fallback for Python versions older than 3.8
    os.environ['PATH'] = os.environ['PATH'] + ';' + x64_release_path + ';' + bin_path

# 4. IMPORT
try:
    import pyopenpose as op
    print("OpenPose imported successfully!")
except ImportError as e:
    print("\n[ERROR] Failed to import pyopenpose.")
    print(f"Looking for library in: {python_openpose_path}")
    print("Troubleshooting checklist:")
    print("1. Go to that folder. Do you see a file ending in .pyd? (e.g. pyopenpose.cp39-win_amd64.pyd)")
    print(f"2. Does the 'cpXX' in that file match your current Python version? (You are using Python {sys.version_info.major}.{sys.version_info.minor})")
    raise e

# --- 2. CONFIGURE OPENPOSE ---
params = dict()
# IMPORTANT: Point this to your 'openpose/models' folder
# This folder should contain subfolders like 'pose/body_25', 'face', etc.
params["model_folder"] = r"C:\Users\Rottem2\Desktop\Rottem\school\project_a\openpose\models"

params["num_gpu"] = 0       # Force CPU usage
params["num_gpu_start"] = 0 
# Now you can use higher resolution because RAM is usually larger than VRAM
params["net_resolution"] = "-1x256"

#params["net_resolution"] = "-1x224"

# Optional: Enable Hand or Face detection if needed for your project
# params["face"] = False
# params["hand"] = False

# --- 3. START THE WRAPPER ---
try:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("OpenPose Wrapper started successfully!")
except Exception as e:
    print(f"Error starting OpenPose: {e}")
    sys.exit(-1)

# video setup
video_path = r"C:\Users\Rottem2\Desktop\Rottem\school\project_a\Cam1(1+2).MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    sys.exit(-1)

target_frame_number = 3100
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)

res, frame = cap.read()

if not res:
    print(f"Error: Could not read frame {target_frame_number} from video.")
    sys.exit(-1)

# Create the specific OpenPose data container
datum = op.Datum()
datum.cvInputData = frame
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# --- 5. INSPECT RESULTS ---
if datum.poseKeypoints is not None:
    print(f"Body Keypoints Shape: {datum.poseKeypoints.shape}")
    print("Coordinates for the first person detected:")
    print(datum.poseKeypoints[0]) 

else:
    print("No body detected in the image.")


# --- REPLACE FROM HERE DOWN ---

window_name = "Video Frame"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

# 1. Define a variable to hold the image we want to show
image_to_show = None

# 2. Try to get the OpenPose output first
if datum.cvOutputData is not None:
    # Check if the image actually has valid dimensions (rows > 0, cols > 0)
    if datum.cvOutputData.shape[0] > 0 and datum.cvOutputData.shape[1] > 0:
        image_to_show = datum.cvOutputData

# 3. If OpenPose failed, fallback to the original frame
if image_to_show is None:
    print("Warning: OpenPose did not return a valid image. Showing original frame.")
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        image_to_show = frame

# 4. If BOTH are bad, we cannot show anything
if image_to_show is not None:
    cv2.imshow(window_name, image_to_show)
    print("Displaying image. Press any key to exit.")
    cv2.waitKey(0)
else:
    print("Error: Both OpenPose output and original frame are empty. Cannot display.")

cv2.destroyAllWindows()