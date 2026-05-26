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
params["net_resolution"] = "-1x160"

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

# --- 4. PROCESS AN IMAGE ---
# Create a dummy image (black screen) if you don't have one ready, 
# or load a real image using cv2.imread("path/to/image.jpg")
image_path = r"C:\Users\Rottem2\Desktop\Rottem\school\project_a\20220711_205110.jpg"
image_to_process = cv2.imread(image_path)
if image_to_process is None:
    print(f"Failed to load image at {image_path}. Using a blank image instead.")


#image_to_process = np.zeros((720, 1280, 3), dtype=np.uint8) 

# Create the specific OpenPose data container
datum = op.Datum()
datum.cvInputData = image_to_process

# Run the processing
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# --- 5. INSPECT RESULTS ---
if datum.poseKeypoints is not None:
    print(f"Body Keypoints Shape: {datum.poseKeypoints.shape}")
    print("Coordinates for the first person detected:")
    print(datum.poseKeypoints[0]) 
    window_name = "OpenPose Result"
    
    # 1. Create a window that allows resizing
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 2. Force the window to open at a specific size (e.g., 800x600)
    cv2.resizeWindow(window_name, 300, 400)
    
    # 3. Show the image
    cv2.imshow(window_name, datum.cvOutputData)
    print("Press 'q' or ESC to exit...")
    
    # 4. Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No body detected in the image.")
