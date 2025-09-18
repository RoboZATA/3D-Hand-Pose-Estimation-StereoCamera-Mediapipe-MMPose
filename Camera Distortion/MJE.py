import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ===== Ground truth distances per video =====
GROUND_TRUTH_VIDEOS = {
    0: 20,  # first video
    1: 30,  # second video
    2: 35,  # third video
    3: 45,  # fourth video
    4: 55,  # fifth video
    5: 65   # sixth video
}

# ===== Load 3D keypoints from JSON =====
def load_keypoints(filepath):
    """Return list of frames, each as numpy array (21,3)."""
    with open(filepath, "r") as f:
        data = json.load(f)
    frames = [np.array(frame["keypoints3d"], dtype=float) for frame in data]
    return frames

# ===== Compute per-frame MPJPE for z-coordinate =====
def compute_z_mpjpe(frames, gt_z):
    """Compute |mean(z) - gt_z| per frame."""
    errors = []
    for kpts in frames:
        valid_kpts = kpts[kpts[:, 2] != -1]  # filter invalid
        if valid_kpts.shape[0] == 0:
            errors.append(np.nan)
            continue
        mean_z = np.mean(valid_kpts[:, 2])
        errors.append(abs(mean_z - gt_z))
    return np.array(errors)

# ===== Main =====
DIST_FILES = sorted(glob.glob("keypoints/*_Dist.json"))
NODIST_FILES = sorted(glob.glob("keypoints/*_NoDist.json"))

all_errors_dist = []
all_errors_nodist = []

for idx, (dist_file, nodist_file) in enumerate(zip(DIST_FILES, NODIST_FILES)):
    gt_z = GROUND_TRUTH_VIDEOS.get(idx, None)
    if gt_z is None:
        continue

    frames_dist = load_keypoints(dist_file)
    frames_nodist = load_keypoints(nodist_file)

    errors_dist = compute_z_mpjpe(frames_dist, gt_z)
    errors_nodist = compute_z_mpjpe(frames_nodist, gt_z)

    all_errors_dist.append(errors_dist)
    all_errors_nodist.append(errors_nodist)

    print(f"{os.path.basename(dist_file)} - Mean MPJPE (z) with Distortion: {np.nanmean(errors_dist):.2f} mm")
    print(f"{os.path.basename(nodist_file)} - Mean MPJPE (z) without Distortion: {np.nanmean(errors_nodist):.2f} mm\n")

# ===== Plot errors across frames =====
plt.figure(figsize=(12,6))
for i, (err_dist, err_nodist) in enumerate(zip(all_errors_dist, all_errors_nodist)):
    plt.plot(err_dist, label=f"Video {i+1} with Distortion", linestyle='-')
    plt.plot(err_nodist, label=f"Video {i+1} without Distortion (modeled)", linestyle='--')

plt.xlabel("Frame")
plt.ylabel("MPJPE (z) [mm]")
plt.title("MPJPE per Frame (Z Coordinate) for Distorted vs. NoDistorted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Compute overall mean MPJPE =====
mean_dist = np.nanmean([np.nanmean(e) for e in all_errors_dist])
mean_nodist = np.nanmean([np.nanmean(e) for e in all_errors_nodist])

print(f"Overall MPJPE (z) with Distortion: {mean_dist:.2f} mm")
print(f"Overall MPJPE (z) without Distortion (modeled): {mean_nodist:.2f} mm")
