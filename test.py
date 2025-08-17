import cv2
import numpy as np
from PIL import Image
import random

# === CONFIG ===
image_path = "h1.jpg"   # Input image
output_path = "low_poly_result.jpg"
resize_width = 700      # More width = more detail
max_edge_points = 3000  # Max points sampled from edges
num_random_points = 1000  # Extra random points

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
edges = cv2.Canny(gray, 50, 150)

# === POINT SELECTION ===

# Edge points
edge_points = np.column_stack(np.where(edges > 0))
if len(edge_points) > max_edge_points:
    np.random.seed(42)
    edge_points = edge_points[np.random.choice(edge_points.shape[0], max_edge_points, replace=False)]

# Multi-scale grid points with jitter
grid_points = []
for grid_size in [5, 15, 30, 60]:  # fine, medium, coarse grids
    for y in range(0, new_height, grid_size):
        for x in range(0, resize_width, grid_size):
            jitter_amount = grid_size // 2
            jittered_y = np.clip(y + random.randint(-jitter_amount, jitter_amount), 0, new_height - 1)
            jittered_x = np.clip(x + random.randint(-jitter_amount, jitter_amount), 0, resize_width - 1)
            grid_points.append([jittered_y, jittered_x])
grid_points = np.array(grid_points)

# Completely random scatter points
random_points = np.column_stack((
    np.random.randint(0, new_height, num_random_points),
    np.random.randint(0, resize_width, num_random_points)
))

# Image corners
corner_points = np.array([[0, 0], [0, new_height - 1],
                          [resize_width - 1, 0], [resize_width - 1, new_height - 1]])

# Combine all points
points = np.concatenate((edge_points, grid_points, random_points, corner_points), axis=0)
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

# === DRAW TRIANGLES WITH AVERAGE COLOR ===
triangles = subdiv.getTriangleList().astype(np.int32)
output = np.zeros_like(small_image)

for t in triangles:
    pts = t.reshape(3, 2)
    pts = np.array([pts], dtype=np.int32)

    if np.any(pts < 0) or np.any(pts[:, :, 0] >= resize_width) or np.any(pts[:, :, 1] >= new_height):
        continue

    # Mask for the triangle
    mask = np.zeros((new_height, resize_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)

    # Average color of pixels inside the triangle
    mean_color = cv2.mean(small_image, mask=mask)[:3]

    # Fill triangle with average color
    cv2.fillConvexPoly(output, pts, mean_color)

# === SAVE AND SHOW ===
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
result = Image.fromarray(output_rgb)
result.save(output_path)
result.show()

print(f"✅ Low poly image saved as: {output_path}")
