import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
import os
from mmpose.apis import MMPoseInferencer
from utils import triangulate_fisheye
from utils import read_camera_parameters, read_rotation_translation

# =========================
# Config
# =========================
DATA_DIR = "data3D"
os.makedirs(DATA_DIR, exist_ok=True)
KEYPOINTS_FILE = os.path.join(DATA_DIR, "keypoints_3d.ndjson")
VIDEO_FILE = os.path.join(DATA_DIR, "output.mp4")

# Original stereo frame size
frame_shape = [1200, 1600]
# Reduced frame size for detection to speed up inference
detect_shape = (640, 480)
plot_interval = 5  # update 3D plot every N frames

# Hand skeleton for visualization
fingers = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}
fingers_colors = {
    "thumb": "orange",
    "index": "black",
    "middle": "green",
    "ring": "blue",
    "pinky": "red",
}

# =========================
# Load MMPose Model
# =========================
inferencer = MMPoseInferencer(
    pose2d="td-hm_litehrnet-w18_8xb32-210e_coco-wholebody-hand-256x256",
    pose2d_weights="https://download.openmmlab.com/mmpose/hand/litehrnet/litehrnet_w18_coco_wholebody_hand_256x256-d6945e6a_20210908.pth"
)

# =========================
# Camera parameters
# =========================
def load_camera_params():
    K_left, D_left = read_camera_parameters(0)
    K_right, D_right = read_camera_parameters(1)
    R0, T0 = read_rotation_translation(0)
    R1, T1 = read_rotation_translation(1)
    R_rel = R0 @ R1.T
    T_rel = T1
    return K_left, D_left, K_right, D_right, R_rel, T_rel

# =========================
# Helpers
# =========================
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
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2],
                color=fingers_colors[name], linewidth=2
            )
    plt.draw()
    plt.pause(0.001)

def extract_keypoints(result):
    """Extract first detected hand's 21 keypoints"""
    kpts = [[-1, -1]] * 21
    preds = result["predictions"][0]  # batch size 1
    if preds:
        hand = preds[0]
        if "keypoints" in hand:
            kpts = np.array(hand["keypoints"]).tolist()
    return kpts

def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    """Draw joints and skeleton"""
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv.circle(frame, (int(x), int(y)), 3, color, -1)
    # Draw skeleton lines
    for name, ids in fingers.items():
        pts = [keypoints[i] for i in ids if keypoints[i][0] > 0]
        if len(pts) == len(ids):
            for i in range(len(pts)-1):
                cv.line(frame, tuple(map(int, pts[i])), tuple(map(int, pts[i+1])), color, 2)
    return frame

# =========================
# Main loop
# =========================
def run(video_path):
    K_left, D_left, K_right, D_right, R, T = load_camera_params()

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Failed to open video file: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30

    out_video = cv.VideoWriter(
        VIDEO_FILE, cv.VideoWriter_fourcc(*"mp4v"),
        fps, (frame_shape[1]*2, frame_shape[0])
    )
    if not out_video.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open.")

    plt.ion()
    global ax
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video reached.")
            break

        width = frame.shape[1] // 2
        frame0 = cv.resize(frame[:, :width], (frame_shape[1], frame_shape[0]))
        frame1 = cv.resize(frame[:, width:], (frame_shape[1], frame_shape[0]))

        # Reduce size for detection
        frame0_small = cv.resize(frame0, detect_shape)
        frame1_small = cv.resize(frame1, detect_shape)

        # Batch inference
        # Frame 0
        results0 = next(inferencer(frame0_small, kpt_thr=0, num_instances=1))
        kpts0 = np.array(extract_keypoints(results0))

        # Frame 1
        results1 = next(inferencer(frame1_small, kpt_thr=0, num_instances=1))
        kpts1 = np.array(extract_keypoints(results1))

        # Scale keypoints back to original frame size
        scale_x = frame_shape[0] / detect_shape[1]
        scale_y = frame_shape[1] / detect_shape[0]
        kpts0[:,0] *= scale_x; kpts0[:,1] *= scale_y
        kpts1[:,0] *= scale_x; kpts1[:,1] *= scale_y

        # Triangulate
        kpts3d = []
        for uv0, uv1 in zip(kpts0, kpts1):
            if uv0[0]==-1 or uv1[0]==-1:
                kpts3d.append(np.array([-1,-1,-1]))
            else:
                kpts3d.append(triangulate_fisheye(uv0, uv1, K_left, D_left, K_right, D_right, R, T))
        kpts3d = np.array(kpts3d)

        append_keypoints_frame(frame_id, kpts3d)

        # Update plot every N frames
        if frame_id % plot_interval == 0:
            update_3d_plot(kpts3d)

        # Draw keypoints and skeleton
        vis0 = draw_keypoints(frame0.copy(), kpts0, color=(0,255,0))
        vis1 = draw_keypoints(frame1.copy(), kpts1, color=(0,0,255))
        combined_frame = np.hstack((vis0, vis1))
        out_video.write(combined_frame.astype(np.uint8))

        # Optional live view
        cv.imshow("Left Camera", vis0)
        cv.imshow("Right Camera", vis1)
        frame_id += 1
        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    out_video.release()
    cv.destroyAllWindows()
    plt.ioff()
    plt.close()

# =========================
# Run
# =========================
if __name__ == "__main__":
    video_path = "test/output.mp4"
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    run(video_path)
