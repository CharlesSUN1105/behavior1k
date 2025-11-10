import sys
import numpy as np
from math import radians
from random import uniform
from b1k.pybullet_utils import (
    init_client,
    load_r1pro,
    step,
    p,
    create_visual_sphere,
    reset_joint_positions,
    get_joint_name_info_map,
    get_wheel_joint_infos,
    load_r1pro_visual,
    PYBULLET_TIME_STEP,
    start_recording_video,
    stop_recording_video,
)
from b1k.models.r1pro.constants import r1pro_init_joint_state
from b1k.planner.single_arm import SingleArmAndTorsoConstraintPlanner


def main():

    T = 30
    side = "left"
    # side = "right"
    planner = SingleArmAndTorsoConstraintPlanner(side, T)

    q0 = [r1pro_init_joint_state[n] for n in planner.joint_names]

    duration = 10.0

    time = np.linspace(0, duration, T)
    Q0 = np.array([q0] * T).T
    # dQ0 = np.gradient(Q0, time, axis=0)
    # ddQ0 = np.gradient(dQ0, time, axis=0)

    config = {
        # Initial guess
        "Q0": Q0,
        # "Q0": np.zeros((planner.serial_chain.dof, T)),
        "dQ0": np.zeros((planner.serial_chain.dof, T)),
        "ddQ0": np.zeros((planner.serial_chain.dof, T)),
        # "Q0": Q0,
        # "dQ0": dQ0,
        # "ddQ0": ddQ0,
        # Parameters
        "duration": duration,
        "q0": q0,
        "w_dQ": 0.01,
        "w_ddQ": 1.0,
        "w_r": 1.0,
        "p_err_max": 0.01 * 0.5,
        "dr": [0, 0, 1],
        "d": 0.7,
        "maintain_gaze": 10.0,
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
    start_point = (
        planner.serial_chain.forward_kinematics(q0)
        .translation()
        .as_vector()
        .toarray()
        .flatten()
    )
    dr_nrm = np.array(config["dr"]) / np.linalg.norm(config["dr"])
    end_point = start_point + config["d"] * dr_nrm

    line_color = [1, 0, 0]  # Red color

    create_visual_sphere(client, 0.04, start_point, (0, 1, 0, 0.4))
    create_visual_sphere(client, 0.04, end_point, (0, 0, 1, 0.4))

    r1pro = load_r1pro(client)
    client.addUserDebugLine(start_point, end_point, line_color, lineWidth=2)
    reset_joint_positions(client, r1pro, r1pro_init_joint_state)

    # Get joint indices
    joint_info_map = get_joint_name_info_map(client, r1pro)
    joint_indices = [joint_info_map[n][0] for n in planner.joint_names]

    # Execute plan
    video = start_recording_video(client, "move_single_arm_constraint.mp4")
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
