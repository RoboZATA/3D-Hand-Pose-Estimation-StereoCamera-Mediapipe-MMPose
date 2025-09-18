import json
import numpy as np
import matplotlib.pyplot as plt

# --- Path to ndjson file ---
filename = 'data3D/keypoints_3d.ndjson'

all_keypoints = []
frame_ids = []

with open(filename, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            frame_id = data.get('frame_id', None)
            keypoints = data.get('keypoints3d', None)

            if frame_id is None or keypoints is None:
                print(f"Skipping line {line_num}: Missing frame_id or keypoints3d")
                continue

            if len(keypoints) != 21:
                print(f"Skipping line {line_num}: Expected 21 keypoints, got {len(keypoints)}")
                continue

            # Convert to numpy array and check shape
            kp_array = np.array(keypoints)
            if kp_array.shape != (21, 3):
                print(f"Skipping line {line_num}: Keypoints shape mismatch {kp_array.shape}")
                continue

            all_keypoints.append(kp_array)
            frame_ids.append(frame_id)

        except json.JSONDecodeError:
            print(f"Skipping line {line_num}: JSON decode error")

if not all_keypoints:
    raise ValueError("No valid keypoints data found.")

# Stack all frames into a 3D numpy array: frames x 21 x 3
keypoints3d = np.stack(all_keypoints, axis=0)
t = np.arange(len(all_keypoints))

# Plotting
plt.figure(figsize=(12, 8))
j=2
# X coordinates
plt.subplot(3, 1, 1)
plt.plot(t, keypoints3d[:, j, 0], label=f'Point {j+1}')
plt.title('X coordinates over time')
plt.xlabel('Frame')
plt.ylabel('X')
plt.grid(True)

# Y coordinates
plt.subplot(3, 1, 2)
plt.plot(t, keypoints3d[:, j, 1])
plt.title('Y coordinates over time')
plt.xlabel('Frame')
plt.ylabel('Y')
plt.grid(True)

# Z coordinates
plt.subplot(3, 1, 3)
plt.plot(t, keypoints3d[:, j, 2])
plt.title('Z coordinates over time')
plt.xlabel('Frame')
plt.ylabel('Z')
plt.grid(True)

plt.tight_layout()
plt.show()
