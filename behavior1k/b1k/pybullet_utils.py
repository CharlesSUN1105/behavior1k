import os
import time
from math import degrees
import pybullet as p
from pathlib import Path
from typing import Callable, Optional
from b1k.paths import r1pro_dir, r1pro_urdf_path
from pybullet_utils.bullet_client import BulletClient
from pybullet_data import getDataPath

PYBULLET_TIME_STEP = 1.0 / 240.0  # default time step

joint_type_map = {
    p.JOINT_REVOLUTE: "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR: "planar",
    p.JOINT_FIXED: "fixed",
}


def init_client(gui: bool = True):
    client = BulletClient(p.GUI if gui else p.DIRECT)
    client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    client.setGravity(0.0, 0.0, -9.81)
    client.loadURDF(os.path.join(getDataPath(), "plane.urdf"), useFixedBase=1)
    return client


def start_recording_video(client: BulletClient, video_path: Path) -> int:
    pth = str(Path(video_path).absolute())
    return client.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, pth)


def stop_recording_video(client: BulletClient, log_id: int):
    client.stopStateLogging(log_id)


def load_r1pro(
    client: BulletClient, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1]
):
    return client.loadURDF(
        str(r1pro_urdf_path.absolute()),
        basePosition=base_position,
        baseOrientation=base_orientation,
        # useFixedBase=1,  # >>TEMP<<
        # flags=p.URDF_USE_IMPLICIT_CYLINDER | p.URDF_MAINTAIN_LINK_ORDER,
    )


def load_r1pro_visual(
    client: BulletClient, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1]
):
    # Load URDF with flags to optimize for visualization
    robot_id = client.loadURDF(
        str(r1pro_urdf_path.absolute()),
        basePosition=base_position,
        baseOrientation=base_orientation,
        flags=client.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        | client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        globalScaling=1.0,
    )

    # Get number of joints
    num_joints = client.getNumJoints(robot_id)

    # Gold color with transparency (RGBA)
    gold_color = [1.0, 0.84, 0.0, 0.3]  # Gold with 30% opacity

    # Configure each link for visualization
    for link_index in range(-1, num_joints):  # -1 for base link
        # Remove collision shapes
        client.setCollisionFilterGroupMask(robot_id, link_index, 0, 0)

        # Change visual properties
        client.changeVisualShape(
            robot_id, link_index, rgbaColor=gold_color, specularColor=[0.5, 0.5, 0.5]
        )

        # Disable physics response
        client.changeDynamics(
            robot_id,
            link_index,
            mass=0,  # Massless
            lateralFriction=0,
            spinningFriction=0,
            rollingFriction=0,
            restitution=0,
            linearDamping=0,
            angularDamping=0,
            contactStiffness=0,
            contactDamping=0,
        )

    # Set overall robot dynamics
    client.changeDynamics(robot_id, -1, mass=0, localInertiaDiagonal=[0, 0, 0])

    # Disable collisions between all links
    for i in range(-1, num_joints):
        for j in range(-1, num_joints):
            if i != j:
                client.setCollisionFilterPair(robot_id, robot_id, i, j, 0)

    return robot_id


def create_visual_sphere(
    client: BulletClient, radius: float, position: list[float], rgba=list[float]
):
    vid = client.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    return client.createMultiBody(
        baseMass=0, baseVisualShapeIndex=vid, basePosition=position
    )


def create_sphere(
    client: BulletClient, radius: float, position: list[float], rgba=list[float]
):
    vid = client.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    cid = client.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    return client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vid,
        baseCollisionShapeIndex=cid,
        basePosition=position,
    )


def get_joint_name_info_map(client: BulletClient, robot: int):
    num_joints = client.getNumJoints(robot)
    joint_name_map: dict[str, int] = {}
    for joint_index in range(num_joints):
        joint_info = client.getJointInfo(robot, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        joint_name_map[joint_name] = joint_info
    return joint_name_map


def in_joint_limit(joint_info, joint_position):
    # Lower limit (minimum position/angle)
    lower_limit = joint_info[8]  # Index 8 for lower limit

    # Upper limit (maximum position/angle)
    upper_limit = joint_info[9]  # Index 9 for upper limit

    return lower_limit <= joint_position <= upper_limit


def reset_joint_positions(
    client: BulletClient, robot: int, joint_positions: dict[str, float]
):
    joint_name_info_map = get_joint_name_info_map(client, robot)
    for joint_name, joint_position in joint_positions.items():
        joint_info = joint_name_info_map[joint_name]
        assert in_joint_limit(
            joint_info, joint_position
        ), f"{joint_name} outside limit [{degrees(joint_info[8])},{degrees(joint_info[9])}], got {degrees(joint_position)}"
        client.resetJointState(robot, joint_info[0], joint_position)


def get_joint_positions(client: BulletClient, robot: int) -> dict[str, float]:
    joint_positions = {}

    for joint_info in get_joint_infos(client, robot, lambda info: True):
        joint_name = joint_info[1].decode("utf-8")
        joint_state = client.getJointState(robot, joint_info[0])
        joint_positions[joint_name] = joint_state[0]  # position
    return joint_positions


def print_joint_types(client: BulletClient, robot: int):
    num_joints = client.getNumJoints(robot)
    joint_infos = []
    num_joint_types = {t: 0 for t in joint_type_map.keys()}
    for joint_index in range(num_joints):
        joint_info = client.getJointInfo(robot, joint_index)
        joint_type = joint_info[2]
        num_joint_types[joint_type] += 1
        print(
            joint_info[0],
            joint_info[1].decode("utf-8"),
            joint_type_map[joint_type],
        )
    print("---")
    for joint_type_id, joint_type_name in joint_type_map.items():
        print(joint_type_name, num_joint_types[joint_type_id])


def get_joint_infos(
    client: BulletClient, robot: int, include: Optional[Callable] = None
):
    num_joints = client.getNumJoints(robot)
    joint_infos = []
    for joint_index in range(num_joints):
        info = client.getJointInfo(robot, joint_index)
        if include and include(info):
            joint_infos.append(info)
    return joint_infos


def get_wheel_joint_infos(client: BulletClient, robot: int):
    include = lambda info: info[1].decode("utf-8").startswith("wheel")
    return get_joint_infos(client, robot, include)


def get_steer_joint_infos(client: BulletClient, robot: int):
    include = lambda info: info[1].decode("utf-8").startswith("steer")
    return get_joint_infos(client, robot, include)


def get_torso_joint_infos(client: BulletClient, robot: int):
    include = lambda info: info[1].decode("utf-8").startswith("torso")
    return get_joint_infos(client, robot, include)


def get_right_arm_joint_infos(client: BulletClient, robot: int):
    include = lambda info: info[1].decode("utf-8").startswith("right_arm")
    return get_joint_infos(client, robot, include)


def get_left_arm_joint_infos(client: BulletClient, robot: int):
    include = lambda info: info[1].decode("utf-8").startswith("left_arm")
    return get_joint_infos(client, robot, include)


def get_right_gripper_finger_joint_infos(client: BulletClient, robot: int):
    include = lambda info: info[1].decode("utf-8").startswith("right_gripper_finger")
    return get_joint_infos(client, robot, include)


def get_left_gripper_finger_joint_infos(client: BulletClient, robot: int):
    include = lambda info: info[1].decode("utf-8").startswith("left_gripper_finger")
    return get_joint_infos(client, robot, include)


def step(client: BulletClient):
    time.sleep(PYBULLET_TIME_STEP)
    client.stepSimulation()


def spin(client: BulletClient):
    while True:
        step(client)
