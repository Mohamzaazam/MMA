#!/usr/bin/env python3
"""
Rotation Diagnostic - Understand the exact transformation needed
to convert CMU ASF/AMC motion to work with human.xml skeleton.

Key insight from C++ analysis:
1. DART ignores BVH offsets - only rotation values matter
2. human.xml joint transforms are IDENTITY - BVH rotations apply directly
3. BVH.cpp applies rotations in channel order: R = Rz * Rx * Ry for ZXY
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_matrix_xyz(x_deg, y_deg, z_deg):
    """ASF convention: X then Y then Z (intrinsic XYZ = extrinsic ZYX)"""
    return R.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True).as_matrix()

def euler_to_matrix_zxy(z_deg, x_deg, y_deg):
    """BVH convention: Z then X then Y (intrinsic ZXY)"""
    return R.from_euler('zxy', [z_deg, x_deg, y_deg], degrees=True).as_matrix()

def matrix_to_euler_zxy(mat):
    """Extract ZXY Euler angles from rotation matrix"""
    return R.from_matrix(mat).as_euler('zxy', degrees=True)

def matrix_to_euler_xyz(mat):
    """Extract XYZ Euler angles from rotation matrix"""
    return R.from_matrix(mat).as_euler('xyz', degrees=True)

print("="*70)
print("ROTATION FRAME ANALYSIS")
print("="*70)

# CMU ASF axis values (from the skeleton)
# rfemur: axis 0 0 -20 (degrees)
# lfemur: axis 0 0 20 (degrees)
print("\n=== CMU ASF Axis Values ===")
print("rfemur axis: (0, 0, -20) degrees - Right leg toed-out 20°")
print("lfemur axis: (0, 0, +20) degrees - Left leg toed-out 20°")

# The axis defines a pre-rotation: R_axis = Rz(-20) for rfemur
R_axis_right = euler_to_matrix_xyz(0, 0, -20)
R_axis_left = euler_to_matrix_xyz(0, 0, 20)

print(f"\nR_axis_right (20° toe-out):\n{R_axis_right}")
print(f"\nR_axis_left (20° toe-out):\n{R_axis_left}")

# Sample CMU motion data (frame 0 from our earlier analysis)
# rfemur motion: rx=-12.62, ry=1.46, rz=20.94
# lfemur motion: rx=-10.64, ry=-2.92, rz=-26.79
print("\n=== Sample CMU Motion (Frame 0) ===")
rfemur_motion = (-12.62, 1.46, 20.94)  # (RX, RY, RZ) in degrees
lfemur_motion = (-10.64, -2.92, -26.79)
print(f"rfemur motion (XYZ): RX={rfemur_motion[0]:.2f}, RY={rfemur_motion[1]:.2f}, RZ={rfemur_motion[2]:.2f}")
print(f"lfemur motion (XYZ): RX={lfemur_motion[0]:.2f}, RY={lfemur_motion[1]:.2f}, RZ={lfemur_motion[2]:.2f}")

# walk_.bvh reference values (frame 0)
print("\n=== walk_.bvh Reference (Frame 0) ===")
walk_right = (4.40, 23.03, -4.44)  # (Z, X, Y) in degrees
walk_left = (0.00, 0.00, 0.00)
print(f"RightUpLeg (ZXY): Z={walk_right[0]:.2f}, X={walk_right[1]:.2f}, Y={walk_right[2]:.2f}")
print(f"LeftUpLeg (ZXY):  Z={walk_left[0]:.2f}, X={walk_left[1]:.2f}, Y={walk_left[2]:.2f}")

# Test different transformation formulas
print("\n" + "="*70)
print("TESTING TRANSFORMATION FORMULAS")
print("="*70)

R_motion_right = euler_to_matrix_xyz(*rfemur_motion)
R_motion_left = euler_to_matrix_xyz(*lfemur_motion)

print("\n--- Formula 1: R_axis * R_motion * R_axis^(-1) (Similarity Transform) ---")
R1_right = R_axis_right @ R_motion_right @ R_axis_right.T
R1_left = R_axis_left @ R_motion_left @ R_axis_left.T
zxy_right = matrix_to_euler_zxy(R1_right)
zxy_left = matrix_to_euler_zxy(R1_left)
print(f"Right leg ZXY: Z={zxy_right[0]:.2f}, X={zxy_right[1]:.2f}, Y={zxy_right[2]:.2f}")
print(f"Left leg ZXY:  Z={zxy_left[0]:.2f}, X={zxy_left[1]:.2f}, Y={zxy_left[2]:.2f}")

print("\n--- Formula 2: R_axis * R_motion (Pre-multiply) ---")
R2_right = R_axis_right @ R_motion_right
R2_left = R_axis_left @ R_motion_left
zxy_right = matrix_to_euler_zxy(R2_right)
zxy_left = matrix_to_euler_zxy(R2_left)
print(f"Right leg ZXY: Z={zxy_right[0]:.2f}, X={zxy_right[1]:.2f}, Y={zxy_right[2]:.2f}")
print(f"Left leg ZXY:  Z={zxy_left[0]:.2f}, X={zxy_left[1]:.2f}, Y={zxy_left[2]:.2f}")

print("\n--- Formula 3: R_motion * R_axis (Post-multiply) ---")
R3_right = R_motion_right @ R_axis_right
R3_left = R_motion_left @ R_axis_left
zxy_right = matrix_to_euler_zxy(R3_right)
zxy_left = matrix_to_euler_zxy(R3_left)
print(f"Right leg ZXY: Z={zxy_right[0]:.2f}, X={zxy_right[1]:.2f}, Y={zxy_right[2]:.2f}")
print(f"Left leg ZXY:  Z={zxy_left[0]:.2f}, X={zxy_left[1]:.2f}, Y={zxy_left[2]:.2f}")

print("\n--- Formula 4: R_motion only (just Euler conversion) ---")
zxy_right = matrix_to_euler_zxy(R_motion_right)
zxy_left = matrix_to_euler_zxy(R_motion_left)
print(f"Right leg ZXY: Z={zxy_right[0]:.2f}, X={zxy_right[1]:.2f}, Y={zxy_right[2]:.2f}")
print(f"Left leg ZXY:  Z={zxy_left[0]:.2f}, X={zxy_left[1]:.2f}, Y={zxy_left[2]:.2f}")

print("\n--- Formula 5: R_axis^(-1) * R_motion (Remove axis pre-rotation) ---")
R5_right = R_axis_right.T @ R_motion_right
R5_left = R_axis_left.T @ R_motion_left
zxy_right = matrix_to_euler_zxy(R5_right)
zxy_left = matrix_to_euler_zxy(R5_left)
print(f"Right leg ZXY: Z={zxy_right[0]:.2f}, X={zxy_right[1]:.2f}, Y={zxy_right[2]:.2f}")
print(f"Left leg ZXY:  Z={zxy_left[0]:.2f}, X={zxy_left[1]:.2f}, Y={zxy_left[2]:.2f}")

print("\n--- Formula 6: R_motion * R_axis^(-1) (Post-remove axis) ---")
R6_right = R_motion_right @ R_axis_right.T
R6_left = R_motion_left @ R_axis_left.T
zxy_right = matrix_to_euler_zxy(R6_right)
zxy_left = matrix_to_euler_zxy(R6_left)
print(f"Right leg ZXY: Z={zxy_right[0]:.2f}, X={zxy_right[1]:.2f}, Y={zxy_right[2]:.2f}")
print(f"Left leg ZXY:  Z={zxy_left[0]:.2f}, X={zxy_left[1]:.2f}, Y={zxy_left[2]:.2f}")

# Now let's think about what human.xml expects
print("\n" + "="*70)
print("ANALYSIS: What does human.xml expect?")
print("="*70)

print("""
From human.xml:
- FemurR Joint transform: IDENTITY (1 0 0 / 0 1 0 / 0 0 1)
- FemurR Body transform: ~180° rotation around X axis

When BVH rotation is (0, 0, 0):
- Joint rotation = Identity
- Leg orientation = determined by Body transform only

From walk_.bvh frame 0:
- LeftUpLeg rotation = (0, 0, 0) → This is the "rest pose"
- RightUpLeg rotation = (4.4, 23, -4.4) → This leg is in motion

This tells us:
- walk_.bvh (0,0,0) = human.xml rest pose
- walk_.bvh was created with this skeleton in mind

For CMU data:
- CMU T-pose has motion = (0, 0, 0) for all joints
- But CMU has axis rotations baked into the skeleton
- axis = (0, 0, ±20) means the leg starts 20° rotated
""")

# Let's verify: what rotation matrix does walk_.bvh frame 0 give for each leg?
print("\n=== Verifying walk_.bvh rotation matrices ===")
R_walk_right = euler_to_matrix_zxy(*walk_right)
R_walk_left = euler_to_matrix_zxy(*walk_left)

print(f"walk_.bvh RightUpLeg rotation matrix:\n{R_walk_right}")
print(f"\nwalk_.bvh LeftUpLeg rotation matrix:\n{R_walk_left}")

# Now the key question: what CMU transformation gives similar matrices?
print("\n" + "="*70)
print("KEY QUESTION: Which formula produces walk_.bvh-like values?")
print("="*70)

print("""
walk_.bvh left leg is (0, 0, 0) - identity rotation.
This means for a "standing still" pose, we want zero rotation.

CMU motion for a similar standing pose should also produce near-zero.
But CMU has axis rotations that add ~20° to each leg.

The ASF model is:
  R_total = R_parent * R_axis * R_motion
  
Where:
  R_axis = pre-rotation defining joint's local coordinate system
  R_motion = the captured motion relative to that local system

BVH model is:
  R_total = R_parent * R_bvh
  
So: R_bvh should equal R_axis * R_motion

BUT if human.xml was designed WITHOUT axis rotations,
then human.xml's rest pose = identity,
while CMU's rest pose = R_axis.

To match, we might need: R_bvh = R_axis * R_motion * R_correction
Where R_correction accounts for rest pose difference.
""")

# Test: For T-pose (motion = 0,0,0), what do we get?
print("\n=== T-pose test: motion = (0, 0, 0) ===")
R_motion_zero = np.eye(3)

print("Formula 1 (Similarity): R_axis * I * R_axis^(-1) = I")
print("Formula 2 (Pre-mult):   R_axis * I = R_axis")
print("Formula 3 (Post-mult):  I * R_axis = R_axis")
print("Formula 4 (Motion only): I")
print("Formula 5 (Remove axis): R_axis^(-1) * I = R_axis^(-1)")
print("Formula 6 (Post-remove): I * R_axis^(-1) = R_axis^(-1)")

print("\nFor CMU T-pose with axis = (0, 0, -20):")
zxy = matrix_to_euler_zxy(np.eye(3))
print(f"  Formula 1/4: ZXY = {zxy}")
zxy = matrix_to_euler_zxy(R_axis_right)
print(f"  Formula 2/3: ZXY = {zxy}")
zxy = matrix_to_euler_zxy(R_axis_right.T)
print(f"  Formula 5/6: ZXY = {zxy}")

print("""
CONCLUSION:
- Formula 1 (Similarity) and Formula 4 (Motion only) give Identity for T-pose
- This matches human.xml expectation (0,0,0 = rest pose)
- But Formula 1 produced crossed legs in the original converter!

The issue might be elsewhere - let me check the actual converter...
""")