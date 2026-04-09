import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks import python
import numpy as np
import time

print(mp.__file__)
print(dir(mp))

base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# Thresholds (tweak to your preference)
SHOULDER_TILT_THRESHOLD = 15       # degrees — acceptable shoulder level difference
FORWARD_LEAN_THRESHOLD = 0.07      # normalized — how far nose is ahead of shoulders
ALERT_COOLDOWN_SECONDS = 10        # minimum seconds between alerts
 
def calculate_angle_from_horizontal(p1, p2):
    """Angle between two points relative to horizontal"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    return abs(np.degrees(np.arctan2(dy, dx)))

def check_posture(landmarks, frame_w, frame_h):
    """landmarks is a list (not .landmark anymore) Returns (is_good: bool, reason: str)."""
    

    def get(idx):
        p = landmarks[idx]
        return p.x * frame_w, p.y * frame_h
    
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
    nose = get(NOSE)
    l_shoulder = get(LEFT_SHOULDER)
    r_shoulder = get(RIGHT_SHOULDER)
    
    # 1. Shoulder tilt
    tilt = calculate_angle_from_horizontal(l_shoulder, r_shoulder)
    if tilt > SHOULDER_TILT_THRESHOLD:
        return False, f"Shoulders uneven ({tilt:.1f})deg tilt"
    
    # 2, Forward lean - nose X vs midpoint of shoulders X (normalized)
    mid_shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
    lean = landmarks[NOSE].x - mid_shoulder_x
    
    if abs(lean) > FORWARD_LEAN_THRESHOLD:
        return False, f"Forward/sideward lean detected ({lean:.2f})"
    
    return True, "Good posture"

def draw_status(frame, is_good, reason):
    color = (0, 200, 0) if is_good else (0, 0, 220)
    label = f"{'OK' if is_good else 'FIX'}: {reason}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    
    last_alert_time = 0
    print("PostureGuard running. Press Q to quit")
    
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        
        timestamp = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            
            is_good, reason = check_posture(landmarks, w, h)
            draw_status(frame, is_good, reason)
            
            now = time.time()
            
            if not is_good and (now - last_alert_time) > ALERT_COOLDOWN_SECONDS:
                print(f"[ALERT] {reason}")
                
        else:
            cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 0), 2)
            
        cv2.imshow("PostureGuard", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
                
    
    print("PostureGuard stopped.")
    
if __name__ == "__main__":
    main()
                
