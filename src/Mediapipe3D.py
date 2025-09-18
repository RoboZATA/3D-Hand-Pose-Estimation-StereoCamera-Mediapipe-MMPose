import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
import os
import time
from utils import triangulate_fisheye
from utils import read_camera_parameters, read_rotation_translation

DATA_DIR = "data3D"
os.makedirs(DATA_DIR, exist_ok=True)
KEYPOINTS_FILE = os.path.join(DATA_DIR, "keypoints_3d.ndjson")
VIDEO_FILE = os.path.join(DATA_DIR, "output.mp4")
frame_shape = [1200, 1600]

# ===== Mediapipe Setup =====
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ===== Plot Setup =====
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fingers = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20]
}
fingers_colors = {
    "thumb": 'orange',
    "index": 'black',
    "middle": 'green',
    "ring": 'blue',
    "pinky": 'red'
}
def load_camera_params():
    K_left, D_left = read_camera_parameters(0)
    K_right, D_right = read_camera_parameters(1)
    R0, T0 = read_rotation_translation(0)
    R1, T1 = read_rotation_translation(1)
    R_rel = R0 @ R1.T
    T_rel = T1
    return K_left, D_left, K_right, D_right, R_rel, T_rel

# ===== Output Helpers =====
def append_keypoints_frame(frame_id, kpts3d):
    entry = {"frame_id": frame_id, "keypoints3d": kpts3d.tolist()}
    with open(KEYPOINTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def update_3d_plot(kpts3d):
    ax.clear()
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-45)
    for name, ids in fingers.items():
        pts = [kpts3d[i] for i in ids if np.all(kpts3d[i] != -1)]
        if len(pts) == len(ids):
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=fingers_colors[name], linewidth=2)
    plt.draw()
    plt.pause(0.001)

# ===== Main Capture =====
def run(stereo_index):
    K_left, D_left, K_right, D_right, R, T = load_camera_params()
    cap = cv.VideoCapture(stereo_index)
    cap.set(3, frame_shape[1] * 2)
    cap.set(4, frame_shape[0])
    fps = cap.get(cv.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    out_video = cv.VideoWriter(VIDEO_FILE, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_shape[1]*2, frame_shape[0]))
    if not out_video.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open. Try changing codec or check permissions.")
    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)
    frame_id = 0
    start_time = time.time()
    while True:
        if time.time() - start_time > 40:
            print("⏱️ 10 seconds elapsed. Stopping capture.")
            break
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame. Exiting.")
            break
        width = frame.shape[1] // 2
        frame0 = cv.resize(frame[:, :width], (frame_shape[1], frame_shape[0]))
        frame1 = cv.resize(frame[:, width:], (frame_shape[1], frame_shape[0]))
        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame0_rgb.flags.writeable = False
        frame1_rgb.flags.writeable = False
        results0 = hands0.process(frame0_rgb)
        results1 = hands1.process(frame1_rgb)
        kpts0 = [[-1, -1]] * 21
        kpts1 = [[-1, -1]] * 21
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    kpts0[i] = [int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])]
                mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    kpts1[i] = [int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])]
                mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        kpts3d = []
        for uv0, uv1 in zip(kpts0, kpts1):
            if uv0[0] == -1 or uv1[0] == -1:
                kpts3d.append(np.array([-1, -1, -1]))
            else:
                point3d = triangulate_fisheye(uv0, uv1, K_left, D_left, K_right, D_right, R, T)
                kpts3d.append(point3d)
        kpts3d = np.array(kpts3d)
        append_keypoints_frame(frame_id, kpts3d)
        update_3d_plot(kpts3d)
        combined_frame = np.hstack((frame0, frame1))
        out_video.write(combined_frame.astype(np.uint8))
        cv.imshow('Left Camera', frame0)
        cv.imshow('Right Camera', frame1)
        frame_id += 1
        if cv.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    out_video.release()
    cv.destroyAllWindows()
    plt.ioff()
    plt.close()

if __name__ == '__main__':
    stereo_index = 0
    if len(sys.argv) > 1:
        stereo_index = int(sys.argv[1])
    run(stereo_index)
