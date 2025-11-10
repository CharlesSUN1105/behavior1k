import sys
import numpy as np
from math import radians
from random import uniform
from b1k.ik import GlobalSingleArmIK
from b1k.pybullet_utils import (
    init_client,
    load_r1pro,
    step,
    p,
    reset_joint_positions,
    get_joint_name_info_map,
    get_wheel_joint_infos,
    load_r1pro_visual,
    PYBULLET_TIME_STEP,
    start_recording_video,
    stop_recording_video,
)
from b1k.models.r1pro.constants import r1pro_init_joint_state
from b1k.planner.single_arm import SingleArmAndTorsoPlanner
from scipy.spatial.transform import Rotation as Rot


def main():

    T = 30
    side = "left"
    # side = "right"
    planner = SingleArmAndTorsoPlanner(side, T)

    q0 = [r1pro_init_joint_state[n] for n in planner.joint_names]

    joint_limit_safety = 0.99
    ik = GlobalSingleArmIK(side, joint_limit_safety=joint_limit_safety)

    sign = 1.0 if side == "left" else -1.0
    pG = [0.7, sign * 0.1, 1.2]

    rGz = np.array([-1.0, 0, 1])
    rGz /= np.linalg.norm(rGz)
    rGy = np.array([0, 1, 0])
    rGx = np.cross(rGy, rGz)

    RG = np.eye(3)
    RG[:, 0] = rGx
    RG[:, 1] = rGy
    RG[:, 2] = rGz

    print(">>>>>>ROTATION GOAL")
    print(RG)
    print(np.linalg.det(RG))
    rG = Rot.from_matrix(RG).as_quat()

    config = {
        "q0": [r1pro_init_joint_state[n] for n in ik.joint_names],
        "pG": pG,
        "rG": rG,
        "p_err_max": 0.01,  # [m]
        "r_err_max": radians(0.1),
        "maintain_gaze": 1000.0,
    }

    ik.reset(config)
    if ik.solve():
        print("IK solved!")
    else:
        print(">>Failed to solve IK<<")
        sys.exit(0)

    qF_dict = ik.get_solution()
    qF = [qF_dict[n] for n in planner.joint_names]

    duration = 10.0

    time = np.linspace(0, duration, T)
    Q0 = np.linspace(q0, qF, T)
    dQ0 = np.gradient(Q0, time, axis=0)
    ddQ0 = np.gradient(dQ0, time, axis=0)

    config = {
        # Initial guess
        # "Q0": np.zeros((planner.serial_chain.dof, T)),
        # "dQ0": np.zeros((planner.serial_chain.dof, T)),
        # "ddQ0": np.zeros((planner.serial_chain.dof, T)),
        "Q0": Q0,
        "dQ0": dQ0,
        "ddQ0": ddQ0,
        # Parameters
        "duration": duration,
        "q0": q0,
        "qF": qF,
        "w_dQ": 0.01,
        "w_ddQ": 1.0,
    }

    planner.reset(config)
    success = planner.solve()
    if success:
        print("solver succeeded!")
    else:
        print("solver failed")
        sys.exit(0)
    qsol = planner.get_solution()

    # Setup simulator
    client = init_client()
    client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1.0],
    )
    r1pro = load_r1pro(client)
    r1pro_visual = load_r1pro_visual(client)
    reset_joint_positions(client, r1pro, r1pro_init_joint_state)
    reset_joint_positions(client, r1pro_visual, r1pro_init_joint_state)
    reset_joint_positions(
        client, r1pro_visual, {n: q for n, q in zip(planner.joint_names, qF)}
    )

    # Get joint indices
    joint_info_map = get_joint_name_info_map(client, r1pro)
    joint_indices = [joint_info_map[n][0] for n in planner.joint_names]

    # Execute plan
    video = start_recording_video(client, "ik_single_arm.mp4")
    t = 0.0
    angle = 0.0
    angle_step = radians(5)
    try:
        while True:
            client.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=angle,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 1.0],
            )
            angle += angle_step
            if t <= duration:
                client.setJointMotorControlArray(
                    bodyUniqueId=r1pro,
                    jointIndices=joint_indices,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[qsol[n](t) for n in planner.joint_names],
                )
            step(client)
            t += PYBULLET_TIME_STEP
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording_video(video)


if __name__ == "__main__":
    main()
