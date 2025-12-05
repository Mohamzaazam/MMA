Key Finding: Rotation Convention Incompatibility

After deep analysis, I found the fundamental issue:

walk.bvh was created by a DIFFERENT tool (likely Maya/MotionBuilder), NOT by any amc2bvh converter
Our amc2bvh applies the C-style axis transform: q_axis * q_motion * q_axis_inv
Walk.bvh has rotations already in world-space without this transform
The C++ renderer (BVH.cpp) correctly parses BVH by name lookup and channel offsets. The issue is that the rotation VALUES themselves are computed differently.

Options:

Skip the axis transform when generating BVH (just output raw rotation values from AMC)
Try inverting the transform (maybe walk.bvh expects q_axis_inv * q_motion * q_axis)
You clarify: Was walk.bvh generated from a specific tool? What rotation convention does your renderer expect?
Which approach would you like me to try first?