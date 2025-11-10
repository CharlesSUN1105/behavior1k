#!/usr/bin/env python3
"""
ICP-based grasp pose adjustment demo.

This script demonstrates how to use vision-based object detection
and ICP to automatically adjust grasp poses.

Usage:
    python demo_icp.py --object fridge --grasp-pose open_fridge_right
    python demo_icp.py --object radio --grasp-pose raise_radio --use-sam
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import TYPE_CHECKING, Optional
import numpy as np
import pybullet as p

from b1k.vision import GraspPoseAdjuster
from b1k.models.r1pro.grasp_poses import grasp_poses
from b1k.models.r1pro.constants import r1pro_init_joint_state
from b1k.pybullet_utils import init_client, reset_joint_positions

if TYPE_CHECKING:
    from pybullet_utils.bullet_client import BulletClient

# Import from existing scripts
import load_objects
import load_r1pro


# ============================================================================
# Object Configuration
# ============================================================================

DEFAULT_CAMERA_POS = [-0.5, 0, 1.5]
DEFAULT_CAMERA_TARGET = [0, 0, 0.9]
DEFAULT_SECOND_CAMERA_POS = [-0.9, -0.5, 2.0]
DEFAULT_SECOND_CAMERA_TARGET = [0, 0, 0.9]
DEFAULT_SECOND_CAMERA_UP = [0.0, 0.0, 1.0]

OBJECT_CONFIGS = {
    "fridge": {
        "mesh_path": load_objects.FRIDGE_OBJ,
        "load_func": load_objects.load_fridge,
        "default_pose": "open_fridge_right",
        "camera_pos": list(DEFAULT_CAMERA_POS),
        "camera_target": list(DEFAULT_CAMERA_TARGET),
        "second_camera_pos": list(DEFAULT_SECOND_CAMERA_POS),
        "second_camera_target": list(DEFAULT_SECOND_CAMERA_TARGET),
        "second_camera_up": list(DEFAULT_SECOND_CAMERA_UP),
    },
    "radio": {
        "mesh_path": load_objects.RADIO_OBJ,
        "load_func": lambda client: load_objects.load_table_and_radio(client)[1],
        "default_pose": "raise_radio",
        "camera_pos": list(DEFAULT_CAMERA_POS),
        "camera_target": list(DEFAULT_CAMERA_TARGET),
        "second_camera_pos": list(DEFAULT_SECOND_CAMERA_POS),
        "second_camera_target": list(DEFAULT_SECOND_CAMERA_TARGET),
        "second_camera_up": list(DEFAULT_SECOND_CAMERA_UP),
    },
    "tray": {
        "mesh_path": load_objects.TRAY_OBJ,
        "load_func": load_objects.load_tray,
        "default_pose": "tray2pan_right",
        "camera_pos": list(DEFAULT_CAMERA_POS),
        "camera_target": list(DEFAULT_CAMERA_TARGET),
        "second_camera_pos": list(DEFAULT_SECOND_CAMERA_POS),
        "second_camera_target": list(DEFAULT_SECOND_CAMERA_TARGET),
        "second_camera_up": list(DEFAULT_SECOND_CAMERA_UP),
    },
    "frying_pan": {
        "mesh_path": load_objects.FRYING_PAN_OBJ,
        "load_func": load_objects.load_frying_pan,
        "default_pose": "frying_pan_left",
        "camera_pos": list(DEFAULT_CAMERA_POS),
        "camera_target": list(DEFAULT_CAMERA_TARGET),
        "second_camera_pos": list(DEFAULT_SECOND_CAMERA_POS),
        "second_camera_target": list(DEFAULT_SECOND_CAMERA_TARGET),
        "second_camera_up": list(DEFAULT_SECOND_CAMERA_UP),
    },
}


CAMERA_LINK_DEFAULTS = {
    "zed_link": {
        
        #   forward_local ≈ [0, -sin(20°),  cos(20°)]
        #   up_local      ≈ [0, -cos(20°), -sin(20°)]
        "forward": np.array([0.0, -0.34202014, 0.93969262]),
        "up": np.array([0.0, -0.93969262, -0.34202014]),
    },
    "right_realsense_link": {
        "forward": np.array([0.0, 0.0, 1.0]),
        "up": np.array([0.0, -1.0, 0.0]),
    },
    "left_realsense_link": {
        "forward": np.array([0.0, 0.0, 1.0]),
        "up": np.array([0.0, -1.0, 0.0]),
    },
}


def get_camera_axes(
    link_name: str,
    user_forward: Optional[tuple[float, float, float]],
    user_up: Optional[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    defaults = CAMERA_LINK_DEFAULTS.get(link_name, {})
    forward = (
        np.array(user_forward, dtype=float)
        if user_forward is not None
        else defaults.get("forward", np.array([1.0, 0.0, 0.0]))
    )
    up = (
        np.array(user_up, dtype=float)
        if user_up is not None
        else defaults.get("up", np.array([0.0, 0.0, 1.0]))
    )
    return forward, up


def suggest_camera_links(client: "BulletClient", robot_id: int) -> None:
    """Print camera-like link names to help user select a valid camera link."""

    keywords = ("camera", "realsense", "zed")
    link_names = []

    num_joints = client.getNumJoints(robot_id)
    for joint_index in range(num_joints):
        joint_info = client.getJointInfo(robot_id, joint_index)
        link_name = joint_info[12].decode("utf-8")
        link_names.append(link_name)

    candidates = [name for name in link_names if any(k in name for k in keywords)]

    if candidates:
        print("  Suggested camera links on this robot:")
        for name in candidates:
            print(f"    - {name}")
    else:
        print("  未检测到包含 camera/realsense/zed 的 link 名称，请使用 --camera-link 指定有效 link。")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Camera axis vector cannot be zero")
    return vec / norm


def compute_camera_view_from_link(
    client: "BulletClient",
    robot_id: int,
    link_name: str,
    forward_axis: np.ndarray,
    up_axis: np.ndarray,
    debug: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    """Compute camera position/target/up vectors from a link pose."""
    link_index = -1
    base_name = client.getBodyInfo(robot_id)[0].decode("utf-8")
    if base_name != link_name:
        num_joints = client.getNumJoints(robot_id)
        for joint_index in range(num_joints):
            joint_info = client.getJointInfo(robot_id, joint_index)
            if joint_info[12].decode("utf-8") == link_name:
                link_index = joint_index
                break
    if base_name != link_name and link_index == -1:
        raise ValueError(f"Link '{link_name}' not found")

    if base_name == link_name:
        position, orientation = client.getBasePositionAndOrientation(robot_id)
    else:
        link_state = client.getLinkState(
            robot_id, link_index, computeForwardKinematics=True
        )
        position = link_state[0]
        orientation = link_state[1]

    rot_matrix = np.array(client.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    forward_world = rot_matrix @ _normalize(forward_axis)
    up_world = rot_matrix @ _normalize(up_axis)
    camera_pos = np.array(position)
    camera_target = camera_pos + forward_world

    if debug:
        # Visualize camera axes in PyBullet for quick sanity check
        scale = 0.3
        client.addUserDebugLine(
            camera_pos,
            camera_pos + forward_world * scale,
            lineColorRGB=[1, 0, 0],
            lifeTime=1.0,
        )
        client.addUserDebugLine(
            camera_pos,
            camera_pos + up_world * scale,
            lineColorRGB=[0, 1, 0],
            lifeTime=1.0,
        )
    return camera_pos.tolist(), camera_target.tolist(), up_world.tolist()


# ============================================================================
# Main Demo
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object",
        type=str,
        required=True,
        choices=list(OBJECT_CONFIGS.keys()),
        help="Target object to grasp",
    )
    parser.add_argument(
        "--grasp-pose",
        type=str,
        default=None,
        help="Grasp pose name (default: use object's default pose)",
    )
    parser.add_argument(
        "--use-sam",
        action="store_true",
        help="Use SAM for segmentation (requires SAM checkpoint)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint file",
    )
    parser.add_argument(
        "--camera-link",
        type=str,
        default=None,
        help=(
            "Optional camera link to align with (default: use preset camera pose)"
        ),
    )
    parser.add_argument(
        "--free-camera",
        action="store_true",
        help="Use preset camera pose instead of camera link pose",
    )
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override free camera position (world frame)",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override free camera target (world frame)",
    )
    parser.add_argument(
        "--camera-forward-axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override camera forward axis in link frame (defaults to per-link config)",
    )
    parser.add_argument(
        "--camera-up-axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override camera up axis in link frame or free camera up vector",
    )
    parser.add_argument(
        "--visualize-camera-axes",
        action="store_true",
        help="Draw debug lines showing camera forward/up directions",
    )
    parser.add_argument(
        "--object-position",
        type=float,
        nargs=2,
        default=None,
        help="Object position [x, y] (default: varies by object)",
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Skip automatic detection, use fixed grasp pose",
    )
    args = parser.parse_args()
    
    # Get object configuration
    obj_config = OBJECT_CONFIGS[args.object]
    grasp_pose_name = args.grasp_pose or obj_config["default_pose"]
    
    if grasp_pose_name not in grasp_poses:
        print(f"❌ Unknown grasp pose: {grasp_pose_name}")
        print(f"Available poses: {', '.join(grasp_poses.keys())}")
        return
    
    # Initialize PyBullet
    print("Initializing PyBullet...")
    client = init_client(gui=True)
    client.setGravity(0, 0, -9.81)
    client.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=40,
        cameraPitch=-30,
        cameraTargetPosition=[0.0, 0.0, 0.5],
    )
    
    # Load ground plane
    client.loadURDF("plane.urdf")
    
    # Load robot
    print("Loading robot...")
    robot_id = load_r1pro.load_r1pro(client, base_position=[-1.0, 0.0, 0.0])
    
    # Load object
    print(f"Loading {args.object}...")
    if args.object_position:
        # Custom position
        if args.object == "fridge":
            object_id = obj_config["load_func"](
                client, fridge_position=tuple(args.object_position)
            )
        elif args.object == "tray":
            object_id = obj_config["load_func"](
                client, tray_position=tuple(args.object_position)
            )
        elif args.object == "frying_pan":
            object_id = obj_config["load_func"](
                client, frying_pan_position=tuple(args.object_position)
            )
        else:
            object_id = obj_config["load_func"](client)
    else:
        # Default position
        object_id = obj_config["load_func"](client)
    
    print(f"✓ {args.object.capitalize()} loaded (ID: {object_id})")
    
    # Stabilize scene
    print("Stabilizing scene...")
    for _ in range(240):
        client.stepSimulation()
    
    # Get base grasp pose
    base_grasp_pose = grasp_poses[grasp_pose_name]
    print(f"\nBase grasp pose: {grasp_pose_name}")
    print(f"  Robot base: {base_grasp_pose['base_position']}")
    
    if not args.no_auto_detect:
        # Initialize vision-based adjuster
        print("\n" + "=" * 60)
        print("Initializing vision-based pose adjuster...")
        print("=" * 60)
        
        try:
            adjuster = GraspPoseAdjuster(
                client=client,
                robot_id=robot_id,
                object_mesh_path=str(obj_config["mesh_path"]),
                camera_link_name=args.camera_link,
                use_sam=args.use_sam,
                sam_checkpoint=args.sam_checkpoint,
            )
            
            # Detect object pose
            print("\nDetecting object pose...")
            if args.camera_link:
                forward_axis, up_axis = get_camera_axes(
                    args.camera_link,
                    args.camera_forward_axis,
                    args.camera_up_axis,
                )
            else:
                forward_axis = up_axis = None

            if args.free_camera or not args.camera_link:
                default_pos = list(DEFAULT_CAMERA_POS)
                default_target = list(DEFAULT_CAMERA_TARGET)
                camera_pos = args.camera_pos or default_pos
                camera_target = args.camera_target or default_target
                if args.camera_up_axis:
                    camera_up = _normalize(np.array(args.camera_up_axis)).tolist()
                else:
                    camera_up = [0.0, 0.0, 1.0]
            else:
                camera_pos, camera_target, camera_up = compute_camera_view_from_link(
                    client,
                    robot_id,
                    args.camera_link,
                    forward_axis,
                    up_axis,
                    debug=args.visualize_camera_axes,
                )
                if args.camera_up_axis:
                    camera_up = _normalize(np.array(args.camera_up_axis)).tolist()
                print(
                    f"  Using camera link '{args.camera_link}' @ pos {camera_pos} looking {camera_target}"
                )
            
            detected_pose, fitness = adjuster.detect_object_pose(
                camera_pos=camera_pos,
                camera_target=camera_target,
                camera_up=tuple(camera_up),
                object_id=object_id,  # Use PyBullet segmentation
            )
            
            print(f"✓ Detection complete")
            print(f"  Fitness score: {fitness:.3f}")
            print(f"  Detected position: {detected_pose[:3, 3]}")
            
            if fitness < 0.3:
                print("⚠️ Warning: Low ICP fitness score, results may be inaccurate")
            
            # Adjust grasp pose
            print("\nAdjusting grasp pose...")
            adjusted_pose = adjuster.adjust_grasp_pose(
                base_grasp_pose=base_grasp_pose,
                detected_object_pose=detected_pose,
                base_object_position=(0.0, 0.0, 0.0),
            )
            
            print(f"✓ Grasp pose adjusted")
            print(f"  Original base: {base_grasp_pose['base_position']}")
            print(f"  Adjusted base: {adjusted_pose['base_position']}")
            
            final_pose = adjusted_pose
            
        except ValueError as e:
            print(f"❌ Vision-based detection failed: {e}")
            if "Link" in str(e):
                suggest_camera_links(client, robot_id)
            print("Falling back to fixed grasp pose...")
            final_pose = base_grasp_pose

        except Exception as e:
            print(f"❌ Vision-based detection failed: {e}")
            print("Falling back to fixed grasp pose...")
            final_pose = base_grasp_pose
    else:
        print("\nUsing fixed grasp pose (no auto-detection)")
        final_pose = base_grasp_pose
    
    # Apply grasp pose
    print("\n" + "=" * 60)
    print("Applying grasp pose...")
    print("=" * 60)
    
    load_r1pro.apply_grasp_pose(
        client, robot_id, grasp_pose_name, skip_base_position=False
    )
    
    print("\n✓ Demo complete!")
    print("Press Ctrl+C to exit...")
    
    # Keep simulation running
    try:
        while True:
            client.stepSimulation()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
