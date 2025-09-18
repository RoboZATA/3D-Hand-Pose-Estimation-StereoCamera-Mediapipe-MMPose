import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# ==============================
# Load keypoints from NDJSON
# ==============================
input_file = 'data3D/keypoints_3d.ndjson'
output_fir = 'data3D/keypoints_3d_fir.ndjson'
output_iir = 'data3D/keypoints_3d_iir.ndjson'

all_keypoints = []
frame_ids = []

with open(input_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            frame_id = data.get('frame_id', None)
            keypoints = data.get('keypoints3d', None)

            if frame_id is None or keypoints is None or len(keypoints) != 21:
                continue

            kp_array = np.array(keypoints)
            if kp_array.shape != (21, 3):
                continue

            all_keypoints.append(kp_array)
            frame_ids.append(frame_id)

        except json.JSONDecodeError:
            continue

if not all_keypoints:
    raise ValueError("No valid keypoints data found.")

keypoints3d = np.stack(all_keypoints, axis=0)  # shape: (frames, 21, 3)
t = np.arange(len(all_keypoints))

# ==============================
# FIR Filter (Gaussian)
# ==============================
window_length = 10
sigma = window_length / 6

fir_filtered = np.zeros_like(keypoints3d)
for j in range(21):
    for d in range(3):
        fir_filtered[:, j, d] = gaussian_filter1d(keypoints3d[:, j, d], sigma=sigma)

# ==============================
# IIR Filter
# y[n] = a*x[n] + (1-a)*y[n-1]
# ==============================
def iir_filter(signal, a=0.2):
    y = np.zeros_like(signal)
    y[0] = signal[0]
    for i in range(1, len(signal)):
        y[i] = a * signal[i] + (1 - a) * y[i - 1]
    return y

iir_filtered = np.zeros_like(keypoints3d)
a = 0.2
for j in range(21):
    for d in range(3):
        iir_filtered[:, j, d] = iir_filter(keypoints3d[:, j, d], a)

# ==============================
# Save FIR Filtered Keypoints
# ==============================
with open(output_fir, 'w') as f_fir:
    for idx, frame_id in enumerate(frame_ids):
        entry = {
            "frame_id": frame_id,
            "keypoints3d": fir_filtered[idx].tolist()
        }
        f_fir.write(json.dumps(entry) + "\n")
print(f"✅ FIR-filtered keypoints saved to: {output_fir}")

# ==============================
# Save IIR Filtered Keypoints
# ==============================
with open(output_iir, 'w') as f_iir:
    for idx, frame_id in enumerate(frame_ids):
        entry = {
            "frame_id": frame_id,
            "keypoints3d": iir_filtered[idx].tolist()
        }
        f_iir.write(json.dumps(entry) + "\n")
print(f"✅ IIR-filtered keypoints saved to: {output_iir}")

# ==============================
# Optional: Visualize Joint 20
# ==============================
joint_index = 20
x_orig = keypoints3d[:, joint_index, 0]
y_orig = keypoints3d[:, joint_index, 1]
z_orig = keypoints3d[:, joint_index, 2]

x_fir = fir_filtered[:, joint_index, 0]
y_fir = fir_filtered[:, joint_index, 1]
z_fir = fir_filtered[:, joint_index, 2]

x_iir = iir_filtered[:, joint_index, 0]
y_iir = iir_filtered[:, joint_index, 1]
z_iir = iir_filtered[:, joint_index, 2]

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, x_orig, label='Original X')
plt.plot(t, x_fir, label='FIR X', linestyle='--')
plt.plot(t, x_iir, label='IIR X', linestyle=':')
plt.title('X Coordinate - Joint 20')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, y_orig, label='Original Y')
plt.plot(t, y_fir, label='FIR Y', linestyle='--')
plt.plot(t, y_iir, label='IIR Y', linestyle=':')
plt.title('Y Coordinate - Joint 20')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, z_orig, label='Original Z')
plt.plot(t, z_fir, label='FIR Z', linestyle='--')
plt.plot(t, z_iir, label='IIR Z', linestyle=':')
plt.title('Z Coordinate - Joint 20')
plt.legend()
plt.xlabel('Frame')
plt.grid(True)

plt.tight_layout()
plt.show()
