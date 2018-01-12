3D Vision
---------
This is tasks:

- [ ] Finish **README.md** doc
- [ ] Code

The step
========

1. [x] Detect 2D points
1. [x] Match 2D points across 2 images
1. [ ] Epipolar geometry
   - If both intrinsic and extrinsic camera parameters are known, reconstruct with projection matrices.
   - If only the intrinsic parameters are known, normalize coordinates and calculate the essential matrix.
   - If neither intrinsic nor extrinsic parameters are known, calculate the
fundamental matrix.
1. [ ] With fundamental or essential matrix, assume P1 = [I 0] and calulate parameters of camera 2.
1. [ ] Triangulate knowing that x1 = P1 * X and x2 = P2 * X.
1. [ ] Bundle adjustment to minimize reprojection errors and refine the 3D coordinates.

