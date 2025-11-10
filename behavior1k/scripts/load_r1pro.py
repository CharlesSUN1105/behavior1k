#!/usr/bin/env python3
"""
R1Pro loading and interactive control script.

Loads the R1Pro robot into a PyBullet scene and optionally spawns objects
such as a table, radio, fridge, tray, or frying pan. Supports applying
predefined grasp poses and running an interactive control loop.

Example usages:
    python load_r1pro.py                      # robot only
    python load_r1pro.py --with-objects       # robot + table + radio
    python load_r1pro.py --with-fridge        # robot + fridge
    python load_r1pro.py --with-tray          # robot + tray
    python load_r1pro.py --with-objects --fix-radio --interactive
    python load_r1pro.py --grasp-pose raise_radio --interactive
    python load_r1pro.py --with-fridge --grasp-pose open_fridge_right --interactive
    python load_r1pro.py --with-tray --grasp-pose tray2pan_right --interactive
    python load_r1pro.py --with-frying-pan --frying-pan-yaw 315 --interactive
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Tuple, List, Dict

import pybullet as p

# Configure module search path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from b1k.pybullet_utils import (  # noqa: E402
    PYBULLET_TIME_STEP,
    init_client,
    load_r1pro,
    get_joint_name_info_map,
    reset_joint_positions,
    spin,
)
from pybullet_utils.bullet_client import BulletClient  # noqa: E402
import load_objects  # noqa: E402
from b1k.models.r1pro.constants import r1pro_init_joint_state  # noqa: E402
from b1k.models.r1pro.grasp_poses import grasp_poses  # noqa: E402


# ============================================================================
# Argument parsing
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-objects",
        action="store_true",
        help="Load the coffee table and radio and place them on the table.",
    )
    parser.add_argument(
        "--with-fridge",
        action="store_true",
        help="Load the fridge (takes precedence over --with-objects if both are set).",
    )
    parser.add_argument(
        "--with-tray",
        action="store_true",
        help="Load the tray at a fixed position (height 0.891 m).",
    )
    parser.add_argument(
        "--tray-position",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="Tray position [x, y] in meters (default [0.0, 0.0]).",
    )
    parser.add_argument(
        "--tray-height",
        type=float,
        default=0.8910,
        help="Tray bottom height in meters (default 0.8910).",
    )
    parser.add_argument(
        "--tray-yaw",
        type=float,
        default=0.0,
        help="Tray yaw in degrees (positive is counter-clockwise around Z).",
    )
    parser.add_argument(
        "--with-frying-pan",
        action="store_true",
        help="Load the frying pan at a fixed position (height 0.932 m).",
    )
    parser.add_argument(
        "--frying-pan-position",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="Frying pan position [x, y] in meters (default [0.0, 0.0]).",
    )
    parser.add_argument(
        "--frying-pan-yaw",
        type=float,
        default=0.0,
        help="Frying pan yaw in degrees (positive is counter-clockwise around Z).",
    )
    parser.add_argument(
        "--fridge-position",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="Fridge position [x, y] in meters (default [0.0, 0.0]).",
    )
    parser.add_argument(
        "--fridge-yaw",
        type=float,
        default=180.0,
        help="Fridge yaw in degrees (positive is counter-clockwise around Z, default 180).",
    )
    parser.add_argument(
        "--table-offset-x",
        type=float,
        default=0.0,
        help="Offset of the table center along the X axis in meters (default 0.0).",
    )
    parser.add_argument(
        "--table-offset-y",
        type=float,
        default=0.0,
        help="Offset of the table center along the Y axis in meters (default 0.0).",
    )
    parser.add_argument(
        "--table-yaw",
        type=float,
        default=0.0,
        help="Table yaw in degrees (positive is counter-clockwise around Z).",
    )
    parser.add_argument(
        "--fix-radio",
        action="store_true",
        help="Keep the radio static on the table (mass = 0).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive sliders for base XY and arm/gripper joints.",
    )
    parser.add_argument(
        "--arm-side",
        choices=["left", "right"],
        default="right",
        help="Arm side to control in interactive mode.",
    )
    parser.add_argument(
        "--base-x-range",
        type=float,
        nargs=2,
        default=(-2.0, 2.0),
        metavar=("MIN", "MAX"),
        help="Slider range for base X in interactive mode.",
    )
    parser.add_argument(
        "--base-y-range",
        type=float,
        nargs=2,
        default=(-2.0, 2.0),
        metavar=("MIN", "MAX"),
        help="Slider range for base Y in interactive mode.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run the simulation for N seconds, then exit (default runs indefinitely).",
    )
    parser.add_argument(
        "--grasp-pose",
        type=str,
        default=None,
        choices=list(grasp_poses.keys()) if grasp_poses else [],
        help=(
            "Apply a predefined grasp pose "
            "(options: horizontal_radio, vertical_radio, raise_radio, "
            "turn_on_radio, tray_right, frying_pan_right, frying_pan_left, "
            "tray2pan_right, fridge_door_left)."
        ),
    )
    return parser.parse_args()




def apply_grasp_pose(
    client: BulletClient,
    robot_id: int,
    grasp_pose_name: str,
    skip_base_position: bool = False,
):
    """
    Apply a predefined grasp pose to the robot.
    
    Args:
        client: PyBullet client
        robot_id: Robot body ID
        grasp_pose_name: Name of the grasp pose to apply
        skip_base_position: If True, skip setting base position (already set during loading)
    """
    if grasp_pose_name not in grasp_poses:
        available_poses = ", ".join(grasp_poses.keys())
        raise ValueError(
            f"Unknown grasp pose: {grasp_pose_name}. "
            f"Available poses: {available_poses}"
        )
    
    pose = grasp_poses[grasp_pose_name]
    print(f"Applying grasp pose: {grasp_pose_name}")
    
    # Set robot base position if not already set
    if not skip_base_position:
        base_pos_xy = pose["base_position"][:2]  # Get x, y from pose
        base_pos = [base_pos_xy[0], base_pos_xy[1], 0.0]  # Set z to 0 (ground level)
        client.resetBasePositionAndOrientation(robot_id, base_pos, [0, 0, 0, 1])
        print(f"  Robot base position: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
    else:
        # Verify current position matches expected position
        current_pos, _ = client.getBasePositionAndOrientation(robot_id)
        expected_pos_xy = pose["base_position"][:2]
        if abs(current_pos[0] - expected_pos_xy[0]) > 0.01 or abs(current_pos[1] - expected_pos_xy[1]) > 0.01:
            print(
                "  ⚠️ Warning: robot base position "
                f"[{current_pos[0]:.3f}, {current_pos[1]:.3f}] does not match "
                f"expected [{expected_pos_xy[0]:.3f}, {expected_pos_xy[1]:.3f}]"
            )
    
    # Set robot joint positions
    reset_joint_positions(client, robot_id, pose["joints"])
    
    # Enable position control for all joints to maintain pose while stabilizing
    joint_info_map = get_joint_name_info_map(client, robot_id)
    for joint_name, joint_position in pose["joints"].items():
        if joint_name in joint_info_map:
            joint_info = joint_info_map[joint_name]
            joint_index = joint_info[0]
            client.setJointMotorControl2(
                robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joint_position,
                force=200.0,
            )
    
    # Step simulation to let robot settle on ground and stabilize
    print("  Stabilizing robot pose...")
    for _ in range(100):
        client.stepSimulation()
        # Continue applying position control to maintain pose
        for joint_name, joint_position in pose["joints"].items():
            if joint_name in joint_info_map:
                joint_info = joint_info_map[joint_name]
                joint_index = joint_info[0]
                client.setJointMotorControl2(
                    robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=joint_position,
                    force=200.0,
                )
    
    print(f"✓ Grasp pose {grasp_pose_name} applied")
    
    # Return the applied joint state so it can be used in interactive mode
    return pose["joints"]


# ============================================================================
# Helper functions
# ============================================================================

def get_link_index_by_name(client: BulletClient, robot_id: int, link_name: str) -> int:
    """
    Resolve a link index from its name.

    In PyBullet:
    - The base link uses index -1; its name comes from getBodyInfo.
    - Other links share indices with their parent joints; their names are
      exposed via getJointInfo()[12] (childFrameName).

    Args:
        client: PyBullet client.
        robot_id: Robot body ID.
        link_name: Desired link name.

    Returns:
        Link index.

    Raises:
        ValueError: If the link name cannot be found.
    """
    num_joints = client.getNumJoints(robot_id)
    
    # Check base link first (index -1)
    body_info = client.getBodyInfo(robot_id)
    if body_info:
        base_link_name = body_info[1].decode("utf-8")  # body name
        if base_link_name == link_name:
            return -1
    
    # Check all other links (joint indices)
    for i in range(num_joints):
        joint_info = client.getJointInfo(robot_id, i)
        child_link_name = joint_info[12].decode("utf-8")  # childFrameName is the link name
        if child_link_name == link_name:
            return i
    
    # If not found, raise error with available link names
    available_links = []
    if body_info:
        available_links.append(body_info[1].decode("utf-8"))
    for i in range(num_joints):
        joint_info = client.getJointInfo(robot_id, i)
        available_links.append(joint_info[12].decode("utf-8"))
    
    raise ValueError(
        f"Link '{link_name}' not found in robot. Available links: {available_links}"
    )


def add_joint_sliders(
    client: BulletClient,
    robot_id: int,
    joint_names: List[str],
    joint_info_map: dict,
    initial_state: dict,
) -> Dict[str, int]:
    """
    Create GUI sliders for the specified joints.

    Args:
        client: PyBullet client.
        robot_id: Robot body ID.
        joint_names: Joint names to expose via sliders.
        joint_info_map: Mapping of joint name to joint info tuple.
        initial_state: Initial joint state values.

    Returns:
        Mapping from joint name to slider ID.
    """
    sliders: Dict[str, int] = {}
    for name in joint_names:
        info = joint_info_map.get(name)
        if not info:
            continue
        joint_type = info[2]
        lower = info[8]
        upper = info[9]
        if lower >= upper:
            if joint_type == p.JOINT_PRISMATIC:
                lower, upper = -0.05, 0.05
            else:
                lower, upper = -math.pi, math.pi
        default = initial_state.get(name, 0.0)
        slider_id = client.addUserDebugParameter(name, lower, upper, default)
        if slider_id < 0:
            print(f"⚠️ Failed to create slider for joint {name}; GUI may be disabled.")
            continue
        sliders[name] = slider_id
        client.setJointMotorControl2(
            robot_id,
            info[0],
            controlMode=p.POSITION_CONTROL,
            targetPosition=default,
            force=200.0,
        )
    return sliders


# ============================================================================
# Interactive control
# ============================================================================

def interactive_control(
    client: BulletClient,
    robot_id: int,
    arm_side: str,
    base_x_range: Tuple[float, float],
    base_y_range: Tuple[float, float],
    initial_joint_state: dict = None,
):
    """
    Interactive control mode with GUI sliders and terminal input.
    
    Args:
        client: PyBullet client
        robot_id: Robot body ID
        arm_side: Which arm to control ("left" or "right")
        base_x_range: Range for base X slider
        base_y_range: Range for base Y slider
        initial_joint_state: Initial joint state to use (if None, uses r1pro_init_joint_state)
    """
    if initial_joint_state is None:
        initial_joint_state = r1pro_init_joint_state
    
    base_pos, base_orn = client.getBasePositionAndOrientation(robot_id)
    fixed_z = base_pos[2]

    base_x_slider = client.addUserDebugParameter(
        "base_x", base_x_range[0], base_x_range[1], base_pos[0]
    )
    base_y_slider = client.addUserDebugParameter(
        "base_y", base_y_range[0], base_y_range[1], base_pos[1]
    )
    if base_x_slider < 0 or base_y_slider < 0:
        print("⚠️ Failed to create base sliders; please ensure the PyBullet GUI is enabled.")
        return

    joint_info_map = get_joint_name_info_map(client, robot_id)

    torso_joint_names = [n for n in initial_joint_state if n.startswith("torso")]
    
    # Add both arms to the interactive control set
    left_arm_joint_names = [f"left_arm_joint{i}" for i in range(1, 8)]
    right_arm_joint_names = [f"right_arm_joint{i}" for i in range(1, 8)]
    left_gripper_joint_names = [
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
    ]
    right_gripper_joint_names = [
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    ]
    
    # Aggregate all controllable joints
    all_joint_names = (
        left_arm_joint_names + left_gripper_joint_names +
        right_arm_joint_names + right_gripper_joint_names
    )

    arm_sliders = add_joint_sliders(
        client,
        robot_id,
        all_joint_names,
        joint_info_map,
        initial_joint_state,
    )

    # Store manual input values coming from the terminal thread
    manual_values = {
        "base_x": None,
        "base_y": None,
    }
    manual_values.update({name: None for name in all_joint_names})

    # Control flag for the input thread
    input_thread_running = threading.Event()
    input_thread_running.set()

    def print_help():
        print("\n" + "=" * 60)
        print("Manual input commands:")
        print("  base_x <value>      - Set base X position (e.g. base_x 1.5)")
        print("  base_y <value>      - Set base Y position (e.g. base_y -0.5)")
        print("  joint <name> <val>  - Set joint target (e.g. joint left_arm_joint1 0.5)")
        print("  reset               - Clear manual overrides and revert to sliders")
        print("  help                - Show this help message")
        print("  quit                - Exit interactive mode")
        print("=" * 60 + "\n")

    def input_handler():
        """Process user input in a background thread."""
        print_help()
        while input_thread_running.is_set():
            try:
                cmd = input("> ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == "quit" or cmd[0] == "exit":
                    input_thread_running.clear()
                    break
                elif cmd[0] == "help":
                    print_help()
                elif cmd[0] == "base_x":
                    if len(cmd) >= 2:
                        try:
                            value = float(cmd[1])
                            if base_x_range[0] <= value <= base_x_range[1]:
                                manual_values["base_x"] = value
                                print(f"✓ Set base_x = {value}")
                            else:
                                print(f"⚠️ Value out of range [{base_x_range[0]}, {base_x_range[1]}]")
                        except ValueError:
                            print("⚠️ Invalid numeric value")
                    else:
                        print("⚠️ Usage: base_x <value>")
                elif cmd[0] == "base_y":
                    if len(cmd) >= 2:
                        try:
                            value = float(cmd[1])
                            if base_y_range[0] <= value <= base_y_range[1]:
                                manual_values["base_y"] = value
                                print(f"✓ Set base_y = {value}")
                            else:
                                print(f"⚠️ Value out of range [{base_y_range[0]}, {base_y_range[1]}]")
                        except ValueError:
                            print("⚠️ Invalid numeric value")
                    else:
                        print("⚠️ Usage: base_y <value>")
                elif cmd[0] == "joint":
                    if len(cmd) >= 3:
                        joint_name = cmd[1]
                        if joint_name in manual_values:
                            try:
                                value = float(cmd[2])
                                manual_values[joint_name] = value
                                print(f"✓ Set {joint_name} = {value}")
                            except ValueError:
                                print("⚠️ Invalid numeric value")
                        else:
                            print(f"⚠️ Unknown joint name: {joint_name}")
                            print(f"Available joints: {', '.join(all_joint_names)}")
                    else:
                        print("⚠️ Usage: joint <name> <value>")
                elif cmd[0] == "reset":
                    for name in manual_values:
                        manual_values[name] = None
                    print("✓ Cleared manual overrides; sliders are in control again")
                else:
                    print(f"⚠️ Unknown command: {cmd[0]}, type help to show instructions")
            except EOFError:
                # Exit when stdin closes
                input_thread_running.clear()
                break
            except Exception as e:
                print(f"⚠️ Input handling error: {e}")

    # Start the input thread
    input_thread = threading.Thread(target=input_handler, daemon=True)
    input_thread.start()

    print("Interactive mode:")
    print("  - Use GUI sliders to tweak parameters")
    print("  - Type commands in the terminal to override values")
    print("  - Press Ctrl+C to exit")

    try:
        while input_thread_running.is_set():
            try:
                # Prefer manual overrides; fall back to slider values
                if manual_values["base_x"] is not None:
                    base_x = manual_values["base_x"]
                else:
                    try:
                        base_x = client.readUserDebugParameter(base_x_slider)
                    except Exception:
                        # Fall back to current base pose if slider read fails
                        base_pos, _ = client.getBasePositionAndOrientation(robot_id)
                        base_x = base_pos[0]

                if manual_values["base_y"] is not None:
                    base_y = manual_values["base_y"]
                else:
                    try:
                        base_y = client.readUserDebugParameter(base_y_slider)
                    except Exception:
                        # Fall back to current base pose if slider read fails
                        base_pos, _ = client.getBasePositionAndOrientation(robot_id)
                        base_y = base_pos[1]
            except Exception as e:
                # If base parameters cannot be read, keep current pose and continue
                print(f"⚠️ Failed to read base parameters: {e}")
                print("Continuing with current base pose...")
                base_pos, _ = client.getBasePositionAndOrientation(robot_id)
                base_x, base_y = base_pos[0], base_pos[1]

            client.resetBasePositionAndOrientation(
                robot_id, [base_x, base_y, fixed_z], base_orn
            )
            client.resetBaseVelocity(robot_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

            for name in torso_joint_names:
                info = joint_info_map.get(name)
                if not info:
                    continue
                # Use initial_joint_state (grasp pose or default) instead of always using default
                target_position = initial_joint_state.get(name, 0.0)
                client.setJointMotorControl2(
                    robot_id,
                    info[0],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_position,
                    force=300.0,
                )

            for name, slider_id in arm_sliders.items():
                info = joint_info_map.get(name)
                if not info:
                    continue
                
                # Prefer manual overrides; fall back to slider values
                if manual_values[name] is not None:
                    target = manual_values[name]
                else:
                    try:
                        target = client.readUserDebugParameter(slider_id)
                    except Exception:
                        # Skip joint if slider cannot be read
                        continue
                
                client.setJointMotorControl2(
                    robot_id,
                    info[0],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=200.0,
                )

            client.stepSimulation()
            time.sleep(PYBULLET_TIME_STEP)
    except KeyboardInterrupt:
        print("\nLeaving interactive mode.")
    finally:
        input_thread_running.clear()


# ============================================================================
# Scene stabilization
# ============================================================================

def stabilize_scene(client: BulletClient, steps: int = 240):
    """
    Step the simulation to let the scene settle.

    Args:
        client: PyBullet client.
        steps: Number of simulation steps (default 240, ~1 second).
    """
    for _ in range(steps):
        client.stepSimulation()
        time.sleep(PYBULLET_TIME_STEP)


# ============================================================================
# Main entry point
# ============================================================================

def main():
    args = parse_args()
    client = init_client(gui=True)
    client.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=40,
        cameraPitch=-30,
        cameraTargetPosition=[0.0, 0.0, 0.5],
    )

    print("Loading R1Pro robot...")
    # Determine robot base position
    # If grasp pose is specified, use its base position; otherwise use default position
    if args.grasp_pose:
        # Use base position from grasp pose
        pose = grasp_poses[args.grasp_pose]
        base_pos_xy = pose["base_position"][:2]  # Get x, y from pose
        robot_base_position = [base_pos_xy[0], base_pos_xy[1], 0.0]  # Set z to 0 (ground level)
        print(
            "  Using grasp pose base position: "
            f"[{robot_base_position[0]:.3f}, {robot_base_position[1]:.3f}, {robot_base_position[2]:.3f}]"
        )
    elif args.with_tray:
        robot_base_position = [-1.042, 0.000, 0.0]  # shift backward to avoid overlapping tray
    else:
        # Default pose: slightly back to avoid objects near the origin
        robot_base_position = [-1.0, 0.0, 0.0]
    
    robot_id = load_r1pro(client, base_position=robot_base_position)
    print(f"✓ Robot ID: {robot_id}")
    
    # Apply grasp pose if specified, and save the joint state for interactive mode
    applied_joint_state = None
    if args.grasp_pose:
        # Robot is already at the correct base position, just apply joint angles
        applied_joint_state = apply_grasp_pose(client, robot_id, args.grasp_pose, skip_base_position=True)
    else:
        # Otherwise use default initial joint state
        reset_joint_positions(client, robot_id, r1pro_init_joint_state)
        applied_joint_state = r1pro_init_joint_state

    loaded_objects = ()
    if args.with_fridge:
        print("Loading fridge ...")
        fridge_position = tuple(args.fridge_position)
        fridge_yaw = math.radians(args.fridge_yaw)
        fridge_id = load_objects.load_fridge(
            client,
            fridge_position=fridge_position,
            fridge_yaw=fridge_yaw,
        )
        loaded_objects = (fridge_id,)
        print(f"✓ Fridge ID: {fridge_id}")
    elif args.with_objects:
        # Load table and radio normally
        print("Loading coffee table and radio ...")
        table_xy = (args.table_offset_x, args.table_offset_y)
        table_yaw = math.radians(args.table_yaw)
        loaded_objects = load_objects.load_table_and_radio(
            client,
            table_xy=table_xy,
            table_yaw=table_yaw,
            fix_radio=args.fix_radio,
        )
        print(f"✓ Object IDs: {loaded_objects}")

    # Tray can be loaded alongside other objects
    if args.with_tray:
        print("Loading tray ...")
        tray_position = tuple(args.tray_position)
        tray_height = args.tray_height
        tray_yaw = math.radians(args.tray_yaw)
        tray_id = load_objects.load_tray(
            client,
            tray_position=tray_position,
            tray_height=tray_height,
            tray_yaw=tray_yaw,
        )
        if isinstance(loaded_objects, tuple):
            loaded_objects = loaded_objects + (tray_id,)
        else:
            loaded_objects = (tray_id,)
        print(f"✓ Tray ID: {tray_id}")

    # Frying pan can be loaded alongside other objects
    if args.with_frying_pan:
        print("Loading frying pan ...")
        frying_pan_position = tuple(args.frying_pan_position)
        frying_pan_yaw = math.radians(args.frying_pan_yaw)
        frying_pan_id = load_objects.load_frying_pan(
            client,
            frying_pan_position=frying_pan_position,
            frying_pan_yaw=frying_pan_yaw,
        )
        if isinstance(loaded_objects, tuple):
            loaded_objects = loaded_objects + (frying_pan_id,)
        else:
            loaded_objects = (frying_pan_id,)
        print(f"✓ Frying pan ID: {frying_pan_id}")

    stabilize_scene(client)

    if args.interactive:
        interactive_control(
            client,
            robot_id=robot_id,
            arm_side=args.arm_side,
            base_x_range=tuple(args.base_x_range),
            base_y_range=tuple(args.base_y_range),
            initial_joint_state=applied_joint_state,
        )
        return

    if args.duration is not None:
        sim_steps = max(1, int(args.duration / PYBULLET_TIME_STEP))
        for _ in range(sim_steps):
            client.stepSimulation()
            time.sleep(PYBULLET_TIME_STEP)
        print(f"Simulation will exit after {args.duration:.2f}s.")
        return

    spin(client)


if __name__ == "__main__":
    main()
