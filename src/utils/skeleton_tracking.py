import math
import numpy as np

# Mapping of body parts for different models
class OpenPoseIndices:
    NECK = 1
    R_SHOULDER = 2
    R_WRIST = 4
    L_SHOULDER = 5
    L_WRIST = 7
    MID_HIP = 8
    R_HIP = 9
    R_KNEE = 10
    R_ANKLE = 11
    L_HIP = 12
    L_KNEE = 13
    L_ANKLE = 14

class HRNetIndices:
    # Based on standard COCO 17 layout inside the HRNet body output
    # Note: COCO doesn't have a explicitly defined "MidHip", usually calculated as halfway between L_HIP and R_HIP
    # These will need to be mapped to the 0-16 indices of your HRNet body keypoints output.
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16
    # Mid-hip can be computed dynamically or assigned if present

def get_dist(p1, p2):
    """
    Calculates Euclidean distance between two keypoints.
    Returns None if one of the points is missing (confidence = 0).
    """
    if p1[2] == 0 or p2[2] == 0:
        return None
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_skeleton_valid(keypoints, model_type="openpose", prev_hip_pos=None, min_conf=0.3, 
                      frame_width=None, frame_height=None, roi=None, edge_margin=20):
    """
    Validates the skeleton based on anatomical proportions and movement.
    Also checks if the skeleton is within the allowed Region of Interest (roi) 
    and away from the camera boundaries.
    
    roi format: (x_min, y_min, x_max, y_max)
    """
    if model_type == "openpose":
        idx = OpenPoseIndices
        
        def has_conf(i):
            return keypoints[i][2] >= min_conf

        # 1. Require MidHip
        if not has_conf(idx.MID_HIP):
            return False
            
        mid_hip_x, mid_hip_y = keypoints[idx.MID_HIP][0], keypoints[idx.MID_HIP][1]

        # Region of Motion & Boundary Checks
        if frame_width is not None and frame_height is not None:
            if mid_hip_x < edge_margin or mid_hip_x > frame_width - edge_margin:
                return False
            if mid_hip_y < edge_margin or mid_hip_y > frame_height - edge_margin:
                return False
                
        if roi is not None:
            if not (roi[0] <= mid_hip_x <= roi[2] and roi[1] <= mid_hip_y <= roi[3]):
                return False

        # 2. Leg completeness
        required_leg_points = [idx.R_HIP, idx.R_KNEE, idx.R_ANKLE, idx.L_HIP, idx.L_KNEE, idx.L_ANKLE]
        confident_leg_points = sum(1 for i in required_leg_points if has_conf(i))

        right_leg_complete = has_conf(idx.R_HIP) and has_conf(idx.R_KNEE) and has_conf(idx.R_ANKLE)
        left_leg_complete = has_conf(idx.L_HIP) and has_conf(idx.L_KNEE) and has_conf(idx.L_ANKLE)
        
        if confident_leg_points < 4 or not (right_leg_complete or left_leg_complete):
            return False

        # 3. Torso Reference Length
        torso_len = get_dist(keypoints[idx.NECK], keypoints[idx.MID_HIP])
        if torso_len is None or torso_len < 10:
            return True 

        # 4. Proportions validation
        r_arm = get_dist(keypoints[idx.R_SHOULDER], keypoints[idx.R_WRIST])
        l_arm = get_dist(keypoints[idx.L_SHOULDER], keypoints[idx.L_WRIST])
        
        if r_arm and r_arm > torso_len * 1.5: return False
        if l_arm and l_arm > torso_len * 1.5: return False

        r_leg = get_dist(keypoints[idx.R_HIP], keypoints[idx.R_ANKLE])
        l_leg = get_dist(keypoints[idx.L_HIP], keypoints[idx.L_ANKLE])
        
        if r_leg and r_leg > torso_len * 2.0: return False
        if l_leg and l_leg > torso_len * 2.0: return False

        # 5. Symmetry validation
        if r_arm and l_arm:
            if r_arm > l_arm * 2.0 or l_arm > r_arm * 2.0:
                return False

        # 6. Velocity validation
        if prev_hip_pos is not None:
            current_hip_x = keypoints[idx.MID_HIP][0]
            velocity = abs(current_hip_x - prev_hip_pos)
            max_allowed_jump = torso_len * 0.5  
            if velocity > max_allowed_jump:
                return False
                
        return True

    elif model_type == "hrnet":
        idx = HRNetIndices
        
        def has_conf(i):
            # Keypoints might be shorter if face/hands aren't concatenated yet, but body is first 17
            return len(keypoints) > i and keypoints[i][2] >= min_conf

        # 1. Require Hips to calculate MidHip
        if not (has_conf(idx.L_HIP) and has_conf(idx.R_HIP)):
            return False
            
        # Calculate virtual mid-hip
        mid_hip_x = (keypoints[idx.L_HIP][0] + keypoints[idx.R_HIP][0]) / 2.0
        mid_hip_y = (keypoints[idx.L_HIP][1] + keypoints[idx.R_HIP][1]) / 2.0
        virtual_mid_hip = (mid_hip_x, mid_hip_y, min(keypoints[idx.L_HIP][2], keypoints[idx.R_HIP][2]))

        # Region of Motion & Boundary Checks
        if frame_width is not None and frame_height is not None:
            if mid_hip_x < edge_margin or mid_hip_x > frame_width - edge_margin:
                return False
            if mid_hip_y < edge_margin or mid_hip_y > frame_height - edge_margin:
                return False
                
        if roi is not None:
            if not (roi[0] <= mid_hip_x <= roi[2] and roi[1] <= mid_hip_y <= roi[3]):
                return False

        # 2. Leg completeness
        required_leg_points = [idx.R_HIP, idx.R_KNEE, idx.R_ANKLE, idx.L_HIP, idx.L_KNEE, idx.L_ANKLE]
        confident_leg_points = sum(1 for i in required_leg_points if has_conf(i))

        right_leg_complete = has_conf(idx.R_HIP) and has_conf(idx.R_KNEE) and has_conf(idx.R_ANKLE)
        left_leg_complete = has_conf(idx.L_HIP) and has_conf(idx.L_KNEE) and has_conf(idx.L_ANKLE)
        
        if confident_leg_points < 4 or not (right_leg_complete or left_leg_complete):
            return False

        # 3. Torso Reference Length (Requires virtual neck from shoulders)
        if not (has_conf(idx.L_SHOULDER) and has_conf(idx.R_SHOULDER)):
            return False
            
        neck_x = (keypoints[idx.L_SHOULDER][0] + keypoints[idx.R_SHOULDER][0]) / 2.0
        neck_y = (keypoints[idx.L_SHOULDER][1] + keypoints[idx.R_SHOULDER][1]) / 2.0
        virtual_neck = (neck_x, neck_y, min(keypoints[idx.L_SHOULDER][2], keypoints[idx.R_SHOULDER][2]))
        
        torso_len = get_dist(virtual_neck, virtual_mid_hip)
        if torso_len is None or torso_len < 10:
            return True 

        # 4. Proportions validation
        r_arm = get_dist(keypoints[idx.R_SHOULDER], keypoints[idx.R_WRIST]) if has_conf(idx.R_WRIST) else None
        l_arm = get_dist(keypoints[idx.L_SHOULDER], keypoints[idx.L_WRIST]) if has_conf(idx.L_WRIST) else None
        
        if r_arm and r_arm > torso_len * 1.5: return False
        if l_arm and l_arm > torso_len * 1.5: return False

        r_leg = get_dist(keypoints[idx.R_HIP], keypoints[idx.R_ANKLE]) if has_conf(idx.R_ANKLE) else None
        l_leg = get_dist(keypoints[idx.L_HIP], keypoints[idx.L_ANKLE]) if has_conf(idx.L_ANKLE) else None
        
        if r_leg and r_leg > torso_len * 2.0: return False
        if l_leg and l_leg > torso_len * 2.0: return False

        # 5. Symmetry validation
        if r_arm and l_arm:
            if r_arm > l_arm * 2.0 or l_arm > r_arm * 2.0:
                 return False

        # 6. Velocity validation
        if prev_hip_pos is not None:
            current_hip_x = virtual_mid_hip[0]
            velocity = abs(current_hip_x - prev_hip_pos)
            max_allowed_jump = torso_len * 0.5  
            if velocity > max_allowed_jump:
                return False

        return True

    return False
