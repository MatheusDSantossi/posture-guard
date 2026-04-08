import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Thresholds (tweak to your preference)
SHOULDER_TILT_THRESHOLD = 15       # degrees — acceptable shoulder level difference
FORWARD_LEAN_THRESHOLD = 0.07      # normalized — how far nose is ahead of shoulders
ALERT_COOLDOWN_SECONDS = 10        # minimum seconds between alerts
 
def calculate_angle_from_horizontal(p1, p2):
    """Angle between two points relative to horizontal"""
    dx = p2[0] - p1[0]
    dy = p2[1] = p1[1]
    
    return abs(np.degrees(np.arctan(dy, dx)))

def check_posture(landmarks, frame_w, frame_h):
    """Returns (is_good: bool, reason: str)."""
    lm = landmarks.landmark

    def get(idx):
        p = lm[idx]
        return p.x * frame_w, p.y * frame_h
    
    nose = get(mp_pose.PoseLandmark.NOSE)
    l_shoulder = get(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shoulder = get(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    
    # 1. Shoulder tilt
    tilt = calculate_angle_from_horizontal(l_shoulder, r_shoulder)
    if tilt > SHOULDER_TILT_THRESHOLD:
        return False, f"Shoulders uneven ({tilt:.1f})deg tilt"
    
    # 2, Forward lean - nose X vs midpoint of shoulders X (normalized)
    mid_shoulder_x = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x + lm[mp_pose.PoseLandmark,RIGHT_SHOULDER].x) / 2
    lean = lm[mp_pose.PoeseLandmark.NOSE].x - mid_shoulder_x
    
    if abs(lean) > FORWARD_LEAN_THRESHOLD:
        return False, f"Forward/sideward lean detected ({lean:.2f})"
    
    return True, "Good posture"

def draw_status(frame, is_good, reason):
    color = (0, 200, 0) if is_good else (0, 0, 220)
    label = f"{'OK' if is_good else 'FIX'}: {reason}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0,7, color, 2)
    
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    
    last_alert_time = 0
    print("PostureGuard running. Press Q to quit")
    
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONECTIONS, mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness = 2),
                )
                is_good, reason = check_posture(results.pose_landmarks, w, h)
                
                draw_status(frame, is_good, reason)
                
                now = time.time()
                
                if not is_good and (now - last_alert_time) > ALERT_COOLDOWN_SECONDS:
                    print(f"[ALERT] {reason}")
                    last_alert_time = now
                else:
                    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(frame, "No person detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 0), 2)
                    
                cv2.imshow("PostureGuard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
    cap.release()
    cv2.destroyAllWindows()
    print("PostureGuard stopped.")
    
if __name__ == "__main__":
    main()
                
