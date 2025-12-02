# cmu_to_mass_converter.py
# Run in MotionBuilder: Python Editor -> Load and Execute

from pyfbsdk import *
from pyfbsdk_additions import *

# Joint name mapping: CMU -> MASS (Character1_*)
JOINT_MAP = {
    "root": "Character1_Hips",
    "lowerback": "Character1_Spine",
    "upperback": "Character1_Spine1",
    "thorax": "Character1_Spine2",
    "lowerneck": "Character1_Neck",
    "upperneck": "Character1_Neck1",
    "head": "Character1_Head",
    "lclavicle": "Character1_LeftShoulder",
    "lhumerus": "Character1_LeftArm",
    "lradius": "Character1_LeftForeArm",
    "lwrist": "Character1_LeftHand",
    "rclavicle": "Character1_RightShoulder",
    "rhumerus": "Character1_RightArm",
    "rradius": "Character1_RightForeArm",
    "rwrist": "Character1_RightHand",
    "lhipjoint": "Character1_LeftUpLeg_helper",  # Will be removed
    "lfemur": "Character1_LeftUpLeg",
    "ltibia": "Character1_LeftLeg",
    "lfoot": "Character1_LeftFoot",
    "ltoes": "Character1_LeftToeBase",
    "rhipjoint": "Character1_RightUpLeg_helper",  # Will be removed
    "rfemur": "Character1_RightUpLeg",
    "rtibia": "Character1_RightLeg",
    "rfoot": "Character1_RightFoot",
    "rtoes": "Character1_RightToeBase",
}

# Joints to remove (intermediate nodes with no animation data)
JOINTS_TO_REMOVE = ["reference", "lhipjoint", "rhipjoint"]

# Joints to skip (fingers, thumbs - not in MASS skeleton)
JOINTS_TO_SKIP = ["lhand", "lfingers", "lthumb", "rhand", "rfingers", "rthumb"]


def find_skeleton_root():
    """Find the skeleton root in the scene."""
    scene = FBSystem().Scene
    for comp in scene.Components:
        if isinstance(comp, FBModelSkeleton):
            if comp.Name.lower() in ["reference", "root"]:
                return comp
    return None


def get_all_skeleton_joints(root):
    """Recursively get all skeleton joints."""
    joints = [root]
    for child in root.Children:
        if isinstance(child, FBModelSkeleton):
            joints.extend(get_all_skeleton_joints(child))
    return joints


def transfer_children(source, target):
    """Move all children from source to target."""
    children_to_move = list(source.Children)
    for child in children_to_move:
        child.Parent = target


def bake_reference_transform(reference_node, root_node):
    """
    Bake the reference node's animation into the root node.
    This handles the nested root issue.
    """
    # Get the time span
    player = FBPlayerControl()
    start_time = FBTime(0, 0, 0, 0)
    stop_time = player.ZoomWindowStop
    
    # Get animation nodes
    ref_trans = reference_node.Translation.GetAnimationNode()
    ref_rot = reference_node.Rotation.GetAnimationNode()
    root_trans = root_node.Translation.GetAnimationNode()
    root_rot = root_node.Rotation.GetAnimationNode()
    
    if ref_trans and root_trans:
        # Create FCurves if they don't exist
        if not root_trans.Nodes[0].FCurve:
            root_trans.Nodes[0].FCurve = FBFCurve()
            root_trans.Nodes[1].FCurve = FBFCurve()
            root_trans.Nodes[2].FCurve = FBFCurve()
        
        # Sample and combine translations
        current_time = start_time
        while current_time <= stop_time:
            ref_val = FBVector3d()
            root_val = FBVector3d()
            reference_node.GetVector(ref_val, FBModelTransformationType.kModelTranslation, True, current_time)
            root_node.GetVector(root_val, FBModelTransformationType.kModelTranslation, True, current_time)
            
            combined = FBVector3d(
                ref_val[0] + root_val[0],
                ref_val[1] + root_val[1],
                ref_val[2] + root_val[2]
            )
            
            root_trans.Nodes[0].FCurve.KeyAdd(current_time, combined[0])
            root_trans.Nodes[1].FCurve.KeyAdd(current_time, combined[1])
            root_trans.Nodes[2].FCurve.KeyAdd(current_time, combined[2])
            
            current_time = FBTime(0, 0, 0, current_time.GetFrame() + 1)
    
    print(f"Baked reference transform into root over {current_time.GetFrame()} frames")


def collapse_intermediate_joint(joint):
    """
    Remove an intermediate joint by reparenting its children.
    Used for lhipjoint/rhipjoint which have no meaningful animation.
    """
    parent = joint.Parent
    if parent:
        transfer_children(joint, parent)
        joint.FBDelete()
        print(f"  Collapsed intermediate joint: {joint.Name}")


def rename_joints():
    """Rename CMU joints to MASS naming convention."""
    scene = FBSystem().Scene
    renamed_count = 0
    
    for comp in scene.Components:
        if isinstance(comp, FBModelSkeleton):
            old_name = comp.Name.lower()
            if old_name in JOINT_MAP:
                new_name = JOINT_MAP[old_name]
                print(f"  Renaming: {comp.Name} -> {new_name}")
                comp.Name = new_name
                renamed_count += 1
    
    return renamed_count


def remove_reference_node():
    """Remove the reference node and promote root."""
    scene = FBSystem().Scene
    reference = None
    root = None
    
    # Find reference and root nodes
    for comp in scene.Components:
        if isinstance(comp, FBModelSkeleton):
            if comp.Name.lower() == "reference":
                reference = comp
            elif comp.Name.lower() == "root":
                root = comp
    
    if reference and root:
        print(f"Found reference node: {reference.Name}")
        print(f"Found root node: {root.Name}")
        
        # Bake reference transform into root
        bake_reference_transform(reference, root)
        
        # Reparent root to scene (or reference's parent)
        root.Parent = reference.Parent
        
        # Delete reference node
        reference.FBDelete()
        print("Reference node removed successfully")
        return True
    elif reference and not root:
        print("WARNING: Found reference but no root node!")
        return False
    else:
        print("No reference node found (already clean)")
        return True


def remove_intermediate_joints():
    """Remove hipjoint nodes (they're just offsets with no animation)."""
    scene = FBSystem().Scene
    joints_to_delete = []
    
    for comp in scene.Components:
        if isinstance(comp, FBModelSkeleton):
            if comp.Name.lower() in ["lhipjoint", "rhipjoint"]:
                joints_to_delete.append(comp)
    
    for joint in joints_to_delete:
        collapse_intermediate_joint(joint)


def verify_hierarchy():
    """Print the current skeleton hierarchy for verification."""
    root = find_skeleton_root()
    if not root:
        print("ERROR: No skeleton root found!")
        return
    
    def print_hierarchy(joint, indent=0):
        print("  " * indent + f"└── {joint.Name}")
        for child in joint.Children:
            if isinstance(child, FBModelSkeleton):
                print_hierarchy(child, indent + 1)
    
    print("\nCurrent Hierarchy:")
    print_hierarchy(root)


def main():
    """Main conversion function."""
    print("=" * 60)
    print("CMU to MASS BVH Converter")
    print("=" * 60)
    
    # Step 1: Remove reference node
    print("\n[1/4] Removing reference node...")
    remove_reference_node()
    
    # Step 2: Remove intermediate hip joints
    print("\n[2/4] Removing intermediate hip joints...")
    remove_intermediate_joints()
    
    # Step 3: Rename joints
    print("\n[3/4] Renaming joints to Character1_* convention...")
    count = rename_joints()
    print(f"  Renamed {count} joints")
    
    # Step 4: Verify
    print("\n[4/4] Verifying hierarchy...")
    verify_hierarchy()
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print("\nNEXT STEPS:")
    print("1. File -> Save As to backup")
    print("2. Export BVH with 'Rotation Only' for non-root joints")
    print("   (See export settings below)")


# Run the script
if __name__ == "__main__":
    main()