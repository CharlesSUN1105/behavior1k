from typing import Any, List, Optional

import numpy as np  # TODO: remove, keep here for type hinting for now

import casadi as cs
import spatial_casadi as sc


def calc_jacobian(
    serial_chain: Any, th: List[float], tool: Optional[sc.Transformation] = None
) -> np.ndarray:
    tool = tool or sc.Transformation.identity()
    # ndof = len(th)
    ndof = th.shape[0]
    j_fl = cs.SX.zeros((6, ndof))
    cur_transform = tool.as_matrix()

    cnt = 0
    for f in reversed(serial_chain._serial_frames):
        if f.joint.joint_type == "revolute":
            cnt += 1
            delta = f.joint.axis.T @ cur_transform[:3, :3]
            d = cs.cross(f.joint.axis, cur_transform[:3, 3]).T @ cur_transform[:3, :3]

            j_fl[:, -cnt] = cs.horzcat(d, delta)
        elif f.joint.joint_type == "prismatic":
            cnt += 1
            j_fl[:3, -cnt] = f.joint.axis @ cur_transform[:3, :3]
        cur_frame_transform = f.get_transform(th[-cnt]).as_matrix()
        cur_transform = cur_frame_transform @ cur_transform

    pose = serial_chain.forward_kinematics(th).as_matrix()
    rotation = pose[:3, :3]
    j_tr = cs.SX.zeros(6, 6)
    j_tr[:3, :3] = rotation
    j_tr[3:, 3:] = rotation
    j_w = j_tr @ j_fl
    return j_w


def calc_jacobian_frames(
    serial_chain: Any,
    th: List[float],
    link_name: str,
    tool: Optional[sc.Transformation] = None,
) -> np.ndarray:
    tool = tool or sc.Transformation.identity()
    ndof = len(th)
    j_fl = cs.SX.zeros(6, ndof)
    cur_transform = tool.as_matrix()

    # select first num_th movable joints
    serial_frames = []
    num_movable_joints = 0
    for serial_frame in serial_chain._serial_frames:
        serial_frames.append(serial_frame)
        if serial_frame.joint.joint_type != "fixed":
            num_movable_joints += 1

        if serial_frame.link.name == link_name:
            break  # found first n joints

    cnt = len(th) - num_movable_joints  # only first num_th joints
    for f in reversed(serial_frames):
        if f.joint.joint_type == "revolute":
            cnt += 1
            delta = f.joint.axis @ cur_transform[:3, :3]
            d = np.cross(f.joint.axis, cur_transform[:3, 3]) @ cur_transform[:3, :3]
            j_fl[:, -cnt] = cs.horzcat(d, delta)
        elif f.joint.joint_type == "prismatic":
            cnt += 1
            j_fl[:3, -cnt] = f.joint.axis @ cur_transform[:3, :3]
        cur_frame_transform = f.get_transform(th[-cnt]).as_matrix()
        cur_transform = cur_frame_transform @ cur_transform

    poses = serial_chain.forward_kinematics(th, end_only=False)
    pose = poses[link_name].as_matrix()

    rotation = pose[:3, :3]
    j_tr = cs.SX.zeros(6, 6)
    j_tr[:3, :3] = rotation
    j_tr[3:, 3:] = rotation
    j_w = j_tr @ j_fl

    return j_w
