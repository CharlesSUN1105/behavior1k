import sys
from b1k.pybullet_utils import init_client, load_r1pro
from b1k.diffik import StableBodyTestDifferentialIK
from b1k.pybullet_utils import PYBULLET_TIME_STEP
from b1k.models.r1pro.constants import r1pro_init_joint_state
from b1k.pybullet_utils import (
    init_client,
    load_r1pro,
    step,
    p,
    reset_joint_positions,
    create_visual_sphere,
    create_sphere,
    get_joint_name_info_map,
    get_wheel_joint_infos,
    get_joint_positions,
    load_r1pro_visual,
    PYBULLET_TIME_STEP,
    start_recording_video,
    stop_recording_video,
)

import numpy as np


def get_q(client, robot, joint_names):
    positions = get_joint_positions(client, robot)
    return np.array([positions[n] for n in joint_names])


def main():

    stable = 1.0
    # stable = 0.0

    diffik = StableBodyTestDifferentialIK(PYBULLET_TIME_STEP)

    fk_left = diffik.left_arm_chain.forward_kinematics(
        [
            r1pro_init_joint_state[n]
            for n in diffik.left_arm_chain.get_joint_parameter_names()
        ]
    )
    pG_left = fk_left.translation().as_vector().toarray().flatten()
    fk_right = diffik.right_arm_chain.forward_kinematics(
        [
            r1pro_init_joint_state[n]
            for n in diffik.right_arm_chain.get_joint_parameter_names()
        ]
    )
    pG_right = fk_right.translation().as_vector().toarray().flatten()

    client = init_client()
    client.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=0,
        cameraPitch=-10,
        cameraTargetPosition=[1.25, 0, 1.0],
    )
    r1pro = load_r1pro(client)
    reset_joint_positions(client, r1pro, r1pro_init_joint_state)

    joint_info_map = get_joint_name_info_map(client, r1pro)
    joint_indices = [joint_info_map[n][0] for n in diffik.joint_names]

    sphere_left = create_visual_sphere(client, 0.1, pG_left, (0, 1, 0, 0.8))
    sphere_right = create_visual_sphere(client, 0.1, pG_right, (0, 0, 1, 0.8))
    unit_quat = (0, 0, 0, 1)

    if stable:
        iden = "stable"
    else:
        iden = "unstable"

    video = start_recording_video(client, f"test_stable_robot_{iden}.mp4")

    v = np.array([1.0, 0.0, 0.0])

    try:
        t = 0.0
        maxt = 1.8
        while True:
            if t >= maxt:
                break

            q = get_q(client, r1pro, diffik.joint_names)

            config = {
                "pG_left": pG_left,
                "pG_right": pG_right,
                "q": q,
                "stable": stable,
            }

            diffik.reset(config)
            if diffik.solve():
                print("--ik solved--")
            else:
                print("oh no!")
                sys.exit(0)

            dq = diffik.get_solution()
            q += PYBULLET_TIME_STEP * dq

            client.setJointMotorControlArray(
                bodyUniqueId=r1pro,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=q.tolist(),
            )

            step(client)
            t += PYBULLET_TIME_STEP
            pG_left += PYBULLET_TIME_STEP * v
            pG_right += PYBULLET_TIME_STEP * v

            client.resetBasePositionAndOrientation(sphere_left, pG_left, unit_quat)
            client.resetBasePositionAndOrientation(sphere_right, pG_right, unit_quat)

            print(">>>", t, "<<<")

    except KeyboardInterrupt:
        pass
    finally:
        stop_recording_video(client, video)


if __name__ == "__main__":
    main()
