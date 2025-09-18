import cv2 as cv
import mediapipe as mp
import numpy as np
import glob
import json
import os
from utils import triangulate_fisheye

# ===== Paths =====
VIDEO_DIR = "data"          # Folder containing mp4 videos
OUTPUT_DIR = "keypoints"    # Folder to save keypoints
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Mediapipe Setup =====
mp_hands = mp.solutions.hands
frame_shape = [1200, 1600]  # height, width per camera

# ===== Camera Parameters =====
def load_camera_params():
    K_left = np.array([
        [1057.6967, 0.0, 935.875523],
        [0.0, 1059.28336, 514.807207],
        [0.0, 0.0, 1.0]
    ])
    D_left = np.array([-0.08316627, -0.01301217, -0.11876649, 0.10841191])

    K_right = np.array([
        [1061.52203, 0.0, 923.44557],
        [0.0, 1063.65634, 489.895954],
        [0.0, 0.0, 1.0]
    ])
    D_right = np.array([-0.0886681, 0.00371924, -0.11013399, 0.07893393])

    R = np.eye(3)
    T = np.array([[-5.996469812], [0.13537893], [0.9930152]])
    return K_left, D_left, K_right, D_right, R, T

# ===== Process a single video =====
def process_video(video_path):
    K_left, D_left, K_right, D_right, R, T = load_camera_params()

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open video {video_path}")
        return

    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)

    results_list = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        width = frame.shape[1] // 2
        frame0 = cv.resize(frame[:, :width], (frame_shape[1], frame_shape[0]))
        frame1 = cv.resize(frame[:, width:], (frame_shape[1], frame_shape[0]))

        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        results0 = hands0.process(frame0_rgb)
        results1 = hands1.process(frame1_rgb)

        kpts0 = [[-1, -1]] * 21
        kpts1 = [[-1, -1]] * 21

        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    kpts0[i] = [int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])]
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    kpts1[i] = [int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])]

        # Triangulate using fisheye
        kpts3d = []
        for uv0, uv1 in zip(kpts0, kpts1):
            if uv0[0] == -1 or uv1[0] == -1:
                kpts3d.append([-1, -1, -1])
            else:
                p3d = triangulate_fisheye(uv0, uv1, K_left, D_left, K_right, D_right, R, T)
                kpts3d.append(p3d.tolist())

        results_list.append({
            "frame_id": frame_id,
            "keypoints3d": kpts3d
        })
        frame_id += 1

    cap.release()

    # Save JSON with "NoDist" label
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_json = os.path.join(OUTPUT_DIR, f"{video_name}_NoDist.json")
    with open(output_json, "w") as f:
        json.dump(results_list, f, indent=2)

    print(f"✅ Saved fisheye keypoints (NoDist) for {video_name}")

# ===== Main =====
if __name__ == "__main__":
    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not video_files:
        raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")
    for video_path in video_files:
        process_video(video_path)
