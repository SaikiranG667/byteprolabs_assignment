# 🇮🇳 Flag Pattern Warping with OpenCV

This project overlays a pattern image onto the white region of a flag using **mesh-based image warping** with OpenCV. It also applies **smooth blending** .

---


### Input:
- `Flag.png` – Image of a flag with a white region
- `Pattern.png` – Pattern image to warp onto the flag

### Output:
- `Output.jpg` – Pattern warped on the flag with a smooth finish and clean black border



---

## 🧠 How It Works

1. **White Region Detection**: Detects white area of the flag using HSV thresholding.
2. **Mesh Triangulation**: Creates a mesh grid on the white region and calculates triangles.
3. **Affine Warping**: Warps the pattern image triangle-by-triangle onto the white region.
4. **Smooth Blending**: Blends the warped pattern with the flag using Gaussian-masked alpha blending.


---


