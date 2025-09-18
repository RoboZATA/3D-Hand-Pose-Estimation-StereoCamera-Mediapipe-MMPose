import cv2 as cv
import mediapipe as mp
import numpy as np
import glob
import json
import os
from utils import DLT

# ===== Paths =====
VIDEO_DIR = "data"
OUTPUT_DIR = "keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Mediapipe Setup =====
mp_hands = mp.solutions.hands
frame_shape = [1200, 1600]

# ===== Camera & Projection Matrices =====
def load_projection_matrices():
    K_left = np.array([
        [1.05769670e+03, 0.0, 9.35875523e+02],
        [0.0, 1.05928336e+03, 5.14807207e+02],
        [0.0, 0.0, 1.0]
    ])
    dist_left = np.array([-0.08316627, -0.01301217, -0.11876649, 0.10841191])

    K_right = np.array([
        [1.06152203e+03, 0.0, 9.23445570e+02],
        [0.0, 1.06365634e+03, 4.89895954e+02],
        [0.0, 0.0, 1.0]
    ])
    dist_right = np.array([-0.0886681, 0.00371924, -0.11013399, 0.07893393])

    R0 = np.eye(3)
    T0 = np.zeros((3, 1))
    R1 = np.eye(3)
    T1 = np.array([[-5.996469812], [0.13537893], [0.9930152]])

    P0 = K_left @ np.hstack((R0, T0))
    P1 = K_right @ np.hstack((R1, T1))
    return K_left, dist_left, K_right, dist_right, P0, P1

# ===== Process single video =====
def process_video(video_path):
    K_left, dist_left, K_right, dist_right, P0, P1 = load_projection_matrices()

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

        frame0 = cv.undistort(frame0, K_left, dist_left)
        frame1 = cv.undistort(frame1, K_right, dist_right)

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

        # Triangulate using DLT
        kpts3d = []
        for uv0, uv1 in zip(kpts0, kpts1):
            if uv0[0] == -1 or uv1[0] == -1:
                kpts3d.append([-1, -1, -1])
            else:
                p3d = DLT(P0, P1, uv0, uv1)
                kpts3d.append(p3d.tolist())

        results_list.append({
            "frame_id": frame_id,
            "keypoints3d": kpts3d
        })
        frame_id += 1

    cap.release()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_json = os.path.join(OUTPUT_DIR, f"{video_name}_Dist.json")
    with open(output_json, "w") as f:
        json.dump(results_list, f, indent=2)

    print(f"✅ Saved DLT keypoints (Dist) for {video_name}")

# ===== Main =====
if __name__ == "__main__":
    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not video_files:
        raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")
    for video_path in video_files:
        process_video(video_path)
