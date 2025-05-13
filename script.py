import cv2
import numpy as np

# --- Load and preprocess the flag image ---
flag = cv2.imread('Flag.png')
if flag is None:
    raise ValueError("Flag image not found!")

flag = cv2.resize(flag, (600, 400))
hsv = cv2.cvtColor(flag, cv2.COLOR_BGR2HSV)

# Mask white region of the flag
lower_white = np.array([0, 0, 150])
upper_white = np.array([0, 0, 254])
white_mask = cv2.inRange(hsv, lower_white, upper_white)

# Clean mask
kernel = np.ones((5, 5), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.GaussianBlur(white_mask, (31, 31), 0)
_, white_mask = cv2.threshold(white_mask, 127, 255, cv2.THRESH_BINARY)

# Get mask bounding box
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_shape = np.zeros_like(white_mask)
cv2.drawContours(mask_shape, contours, -1, 255, -1)

x, y, w, h = cv2.boundingRect(mask_shape)
roi_mask = mask_shape[y:y+h, x:x+w].astype(np.float32) / 255.0
flag_roi = flag[y:y+h, x:x+w].copy()

# --- Load and resize the pattern image ---
pattern = cv2.imread('Pattern.png')
if pattern is None:
    raise ValueError("Pattern image not found!")

pattern = cv2.resize(pattern, (w, h))

# --- Create a mesh and triangulate it ---
step = 3
ys, xs = np.where(roi_mask > 0.1)
sample_points = np.array([[x_, y_] for x_, y_ in zip(xs, ys)][::step])

if len(sample_points) < 3:
    raise ValueError("Not enough points to build a mesh.")

src_points = sample_points.astype(np.float32)
src_points[:, 0] = (src_points[:, 0] / w) * pattern.shape[1]
src_points[:, 1] = (src_points[:, 1] / h) * pattern.shape[0]
dst_points = sample_points.astype(np.float32)

triangles = []
cols = w // step
for i in range(len(dst_points) - 1):
    if (i + 1) % cols != 0 and i + cols < len(dst_points):
        triangles.append([i, i + 1, i + cols])
        if i + cols + 1 < len(dst_points):
            triangles.append([i + 1, i + cols + 1, i + cols])

# ---Warp pattern using affine transforms ---
warped_pattern = np.zeros_like(flag_roi)
warped_mask = np.zeros_like(roi_mask)

for tri_indices in triangles:
    if max(tri_indices) >= len(src_points):
        continue

    src_tri = src_points[tri_indices]
    dst_tri = dst_points[tri_indices]

    M = cv2.getAffineTransform(src_tri, dst_tri)
    src_rect = cv2.boundingRect(src_tri)
    dst_rect = cv2.boundingRect(dst_tri)

    src_loc = src_tri - src_rect[:2]
    dst_loc = dst_tri - dst_rect[:2]

    src_crop = pattern[src_rect[1]:src_rect[1]+src_rect[3], src_rect[0]:src_rect[0]+src_rect[2]]
    warped_patch = cv2.warpAffine(src_crop, M, (dst_rect[2], dst_rect[3]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask_tri = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask_tri, np.int32(dst_loc), 255)
    mask_3c = mask_tri[..., None].astype(bool)

    roi_pattern = warped_pattern[dst_rect[1]:dst_rect[1]+dst_rect[3],
                                 dst_rect[0]:dst_rect[0]+dst_rect[2]]
    np.copyto(roi_pattern, warped_patch, where=mask_3c)

    roi_warped_mask = warped_mask[dst_rect[1]:dst_rect[1]+dst_rect[3],
                                  dst_rect[0]:dst_rect[0]+dst_rect[2]]
    roi_warped_mask[mask_tri > 0] = 1.0

# ---Blend pattern with original flag region ---
alpha_blur = cv2.GaussianBlur(warped_mask, (11, 11), 0)
beta = 1.0

blended = (
    warped_pattern.astype(np.float32) * (alpha_blur * beta)[..., None] +
    flag_roi.astype(np.float32) * (1 - alpha_blur * beta)[..., None]
).astype(np.uint8)

roi_mask_3ch = roi_mask[..., None]
final_roi = (
    blended.astype(np.float32) * roi_mask_3ch +
    flag_roi.astype(np.float32) * (1 - roi_mask_3ch)
).astype(np.uint8)

# --- Paste back the warped region into the flag ---
output = flag.copy()
output[y:y+h, x:x+w] = final_roi

# ---Draw a black border on the white region ---
cv2.drawContours(output, contours, -1, (0, 0, 0), 1)

# --- Step 8: Display and Save ---
cv2.imshow("Output", output)
cv2.imwrite("Output.jpg", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
