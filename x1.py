import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

# === CONFIG ===
image_path = "c1.jpg"  # Input image path
output_path = "lowpoly_result.jpg"  # Output path

# === LOAD IMAGE ===
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Convert to PIL for saturation adjustment
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Slightly increase saturation
enhancer = ImageEnhance.Color(pil_img)
pil_img = enhancer.enhance(1.2)  # 1.0 = original, >1 = more saturation

# Back to OpenCV format
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# === Generate Random Points ===
num_points = 500  # Number of points (controls detail level)
points = []

# Add border points
for x in [0, w - 1]:
    for y in range(0, h, h // 10):
        points.append((x, y))
for y in [0, h - 1]:
    for x in range(0, w, w // 10):
        points.append((x, y))

# Add random points (mixed density)
for _ in range(num_points):
    x = random.randint(0, w - 1)
    y = random.randint(0, h - 1)
    points.append((x, y))

# === Delaunay Triangulation ===
subdiv = cv2.Subdiv2D((0, 0, w, h))
for p in points:
    subdiv.insert((int(p[0]), int(p[1])))

triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)

# === Draw Triangles ===
output = np.zeros_like(img)

for t in triangles:
    pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

    # Check if triangle lies inside image
    if all(0 <= px < w and 0 <= py < h for px, py in pts):
        pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))

        # Get average color from original image
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_array, 255)
        mean_color = cv2.mean(img, mask=mask)[:3]
        color = tuple(map(int, mean_color))

        # Draw triangle
        cv2.fillConvexPoly(output, pts_array, color)

# === Save & Show ===
cv2.imwrite(output_path, output)
cv2.imshow("Low Poly Art", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
