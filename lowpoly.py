# import cv2
# import numpy as np
# from PIL import Image
# import random
#
# # === CONFIG ===
# image_path = "m1.jpg"  # Your input image
# output_path = "low_poly_result.jpg"
# resize_width = 700  # Increase for more detail, adjust based on speed
# max_points = 5000   # Increase number of points (triangles) for more detail
#
# # === LOAD IMAGE ===
# image = Image.open(image_path).convert("RGB")
# image_cv = np.array(image)
# image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
#
# orig_height, orig_width = image_cv.shape[:2]
#
# # === RESIZE IMAGE ===
# new_height = int(resize_width * orig_height / orig_width)
# small_image = cv2.resize(image_cv, (resize_width, new_height))
#
# # === EDGE DETECTION ===
# gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds to detect more edges
#
# # === POINT SELECTION ===
#
# # Edge points (all edges, then randomly sample)
# edge_points = np.column_stack(np.where(edges > 0))
# if len(edge_points) > max_points // 2:
#     np.random.seed(42)
#     edge_points = edge_points[np.random.choice(edge_points.shape[0], max_points // 2, replace=False)]
#
# # Multiple grids with different sizes for varied triangle sizes
# grid_points = []
# for grid_size in [10, 20, 40]:  # mix fine, medium and coarse grids
#     for y in range(0, new_height, grid_size):
#         for x in range(0, resize_width, grid_size):
#             grid_points.append([y, x])
# grid_points = np.array(grid_points)
#
# # Random jittered points around grid points to add variation
# random_jitter_points = []
# jitter_amount = 5  # pixels
# for point in grid_points:
#     jittered_y = np.clip(point[0] + random.randint(-jitter_amount, jitter_amount), 0, new_height - 1)
#     jittered_x = np.clip(point[1] + random.randint(-jitter_amount, jitter_amount), 0, resize_width - 1)
#     random_jitter_points.append([jittered_y, jittered_x])
# random_jitter_points = np.array(random_jitter_points)
#
# # Corners of image
# corner_points = np.array([[0, 0], [0, new_height - 1], [resize_width - 1, 0], [resize_width - 1, new_height - 1]])
#
# # Combine all points
# points = np.concatenate((edge_points, grid_points, random_jitter_points, corner_points), axis=0)
#
# # Remove duplicate points (optional but cleaner)
# points = np.unique(points, axis=0)
#
# # === DELAUNAY TRIANGULATION ===
# subdiv = cv2.Subdiv2D((0, 0, resize_width, new_height))
# for p in points:
#     x, y = int(p[1]), int(p[0])
#     if 0 <= x < resize_width and 0 <= y < new_height:
#         try:
#             subdiv.insert((x, y))
#         except cv2.error:
#             pass
#
# # === DRAW TRIANGLES WITH ORIGINAL COLORS ===
# triangles = subdiv.getTriangleList().astype(np.int32)
# output = np.zeros_like(small_image)
#
# for t in triangles:
#     pts = t.reshape(3, 2)
#     pts = np.array([pts], dtype=np.int32)
#
#     if np.any(pts < 0) or np.any(pts[:, 0] >= resize_width) or np.any(pts[:, 1] >= new_height):
#         continue
#
#     # Centroid in resized image coords
#     cx = int(np.mean(pts[0][:, 0]))
#     cy = int(np.mean(pts[0][:, 1]))
#
#     # Map centroid to original image coords
#     orig_cx = int(cx * orig_width / resize_width)
#     orig_cy = int(cy * orig_height / new_height)
#     orig_cx = np.clip(orig_cx, 0, orig_width - 1)
#     orig_cy = np.clip(orig_cy, 0, orig_height - 1)
#
#     color = image_cv[orig_cy, orig_cx].tolist()
#
#     cv2.fillConvexPoly(output, pts, color)
#
# # === SAVE AND SHOW ===
# output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# result = Image.fromarray(output_rgb)
# result.save(output_path)
# result.show()
#
# print(f"✅ Low poly image saved as: {output_path}")


#  Code

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

# === CONFIG ===
image_path = "f2.jpg"     # Input image
output_path = "low_poly_result.png"  # Output image
num_points = 12000         # Number of points (controls detail)

# === Load image ===
img = cv2.imread(image_path)
h, w, _ = img.shape

# === Preprocess for detail detection ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# Pick points from edges + random points
points = []

# More points in detailed areas (edges)
edge_points = np.column_stack(np.where(edges > 0))
for (y, x) in edge_points[::max(1, len(edge_points)//(num_points//2))]:
    points.append((x, y))

# Add some random points for variety
for _ in range(num_points//2):
    points.append((random.randint(0, w-1), random.randint(0, h-1)))

# Add image corners
points.extend([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])

# === Delaunay Triangulation ===
rect = (0, 0, w, h)
subdiv = cv2.Subdiv2D(rect)
for p in points:
    subdiv.insert((int(p[0]), int(p[1])))

triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)

# === Draw triangles ===
output = np.zeros_like(img)

for t in triangles:
    pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
    pts = np.array(pts, np.int32)

    if np.any(pts[:, 0] < 0) or np.any(pts[:, 1] < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] >= h):
        continue

    # Mask for triangle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)

    # Average color inside triangle
    mean_color = cv2.mean(img, mask=mask)
    color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

    # Fill triangle
    cv2.fillConvexPoly(output, pts, color)

# === Convert to PIL for final enhancement ===
pil_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

# Slightly boost saturation for better look
enhancer = ImageEnhance.Color(pil_img)
final_img = enhancer.enhance(1.2)  # 1.0 = original, >1 more saturation

# Save and show result
final_img.save(output_path)
final_img.show()

#