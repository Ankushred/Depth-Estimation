import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Load depth map (grayscale image, 16-bit if available)
depth_map = cv2.imread("depth_map_grayscale.png", cv2.IMREAD_UNCHANGED)

# Camera Intrinsics (Modify based on your camera)
fx, fy = 500, 500  # Focal length in pixels
cx, cy = depth_map.shape[1] / 2, depth_map.shape[0] / 2  # Principal point

# Generate 3D point cloud
height, width = depth_map.shape
points = []

for v in range(height):
    for u in range(width):
        Z = depth_map[v, u] / 1000.0  # Convert to meters (if needed)
        if Z == 0: continue  # Ignore zero-depth pixels
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points.append((X, Y, Z))

# Convert to NumPy array
points = np.array(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2])
plt.show()
