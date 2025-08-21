# import cv2
# import numpy as np
# from PIL import Image
# import random



import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

# === CONFIG ===
image_path = "x1.jpg"     # Input image
output_path = "low_poly_result.png"  # Output image
num_points = 50000         # Number of points (controls detail)

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

