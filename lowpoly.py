import cv2
import numpy as np
from PIL import Image
import random

# === CONFIG ===
image_path = "he1.jpg"  # Your input image
output_path = "low_poly_result.jpg"
resize_width = 700  # Increase for more detail, adjust based on speed
max_points = 5000   # Increase number of points (triangles) for more detail

# === LOAD IMAGE ===
image = Image.open(image_path).convert("RGB")
image_cv = np.array(image)
image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

orig_height, orig_width = image_cv.shape[:2]

# === RESIZE IMAGE ===
new_height = int(resize_width * orig_height / orig_width)
small_image = cv2.resize(image_cv, (resize_width, new_height))

# === EDGE DETECTION ===
gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds to detect more edges

# === POINT SELECTION ===

# Edge points (all edges, then randomly sample)
edge_points = np.column_stack(np.where(edges > 0))
if len(edge_points) > max_points // 2:
    np.random.seed(42)
    edge_points = edge_points[np.random.choice(edge_points.shape[0], max_points // 2, replace=False)]

# Multiple grids with different sizes for varied triangle sizes
grid_points = []
for grid_size in [10, 20, 40]:  # mix fine, medium and coarse grids
    for y in range(0, new_height, grid_size):
        for x in range(0, resize_width, grid_size):
            grid_points.append([y, x])
grid_points = np.array(grid_points)

# Random jittered points around grid points to add variation
random_jitter_points = []
jitter_amount = 5  # pixels
for point in grid_points:
    jittered_y = np.clip(point[0] + random.randint(-jitter_amount, jitter_amount), 0, new_height - 1)
    jittered_x = np.clip(point[1] + random.randint(-jitter_amount, jitter_amount), 0, resize_width - 1)
    random_jitter_points.append([jittered_y, jittered_x])
random_jitter_points = np.array(random_jitter_points)

# Corners of image
corner_points = np.array([[0, 0], [0, new_height - 1], [resize_width - 1, 0], [resize_width - 1, new_height - 1]])

# Combine all points
points = np.concatenate((edge_points, grid_points, random_jitter_points, corner_points), axis=0)

# Remove duplicate points (optional but cleaner)
points = np.unique(points, axis=0)

# === DELAUNAY TRIANGULATION ===
subdiv = cv2.Subdiv2D((0, 0, resize_width, new_height))
for p in points:
    x, y = int(p[1]), int(p[0])
    if 0 <= x < resize_width and 0 <= y < new_height:
        try:
            subdiv.insert((x, y))
        except cv2.error:
            pass

# === DRAW TRIANGLES WITH ORIGINAL COLORS ===
triangles = subdiv.getTriangleList().astype(np.int32)
output = np.zeros_like(small_image)

for t in triangles:
    pts = t.reshape(3, 2)
    pts = np.array([pts], dtype=np.int32)

    if np.any(pts < 0) or np.any(pts[:, 0] >= resize_width) or np.any(pts[:, 1] >= new_height):
        continue

    # Centroid in resized image coords
    cx = int(np.mean(pts[0][:, 0]))
    cy = int(np.mean(pts[0][:, 1]))

    # Map centroid to original image coords
    orig_cx = int(cx * orig_width / resize_width)
    orig_cy = int(cy * orig_height / new_height)
    orig_cx = np.clip(orig_cx, 0, orig_width - 1)
    orig_cy = np.clip(orig_cy, 0, orig_height - 1)

    color = image_cv[orig_cy, orig_cx].tolist()

    cv2.fillConvexPoly(output, pts, color)

# === SAVE AND SHOW ===
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
result = Image.fromarray(output_rgb)
result.save(output_path)
result.show()

print(f"✅ Low poly image saved as: {output_path}")

