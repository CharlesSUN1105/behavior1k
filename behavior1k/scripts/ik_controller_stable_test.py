import sys
from b1k.pybullet_utils import init_client, load_r1pro
from b1k.ik import DualArmIKController
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

from figure_eight import FigureEight
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=1000)


class Goal:

    def __init__(self, sign1=1.0, sign2=1.0, a1=0.04, a2=0.2):
        self.sign1 = sign1
        self.sign2 = sign2
        self.fig8 = FigureEight(a1, a2, 0.12, 0.1)

    # def __call__(self, t):
    #     # dy, dz = self.fig8.velocity(t).tolist()
    #     # r = self.sign1 * 2 * np.sin(t)
    #     # return np.array([0.0, self.sign1 * dy, self.sign2 * dz, r, 0, 0.0])
    #     # y = 0.1*np.sin(t)
    #     y = 2.0 * np.pi * 0.5 * 0.5 * np.cos(2.0 * np.pi * 0.5 * t)
    #     z = 2 * np.pi * 0.1 * 0.2 * np.cos(2 * np.pi * 0.1 * t)
    #     # return np.array([0, y, self.sign1 * z, 0, 0, 0])
    #     return np.array([0, y, self.sign1 * z, 0, 0, 0])

    def get_position(self, t):
        # y, z = self.fig8.position(t).tolist()
        # return np.array([0.0, self.sign1 * y, self.sign2 * z])
        lo = 0.0
        hi = 1.25
        alpha = (-1.0 + np.sin(t * np.pi * 2.0 * 0.1)) * 0.5

        x = (1 - alpha) * lo + alpha * hi

        return np.array([-x, 0, 0])


def get_q(client, robot, joint_names, return_dict=False):
    positions = get_joint_positions(client, robot)
    if return_dict:
        return {n: position[n] for n in joint_names}
    else:
        return np.array([positions[n] for n in joint_names])


def main():
    ik = DualArmIKController(PYBULLET_TIME_STEP)

    goal_left = Goal(a1=0.05, sign2=-1.0)
    goal_right = Goal(sign1=-1.0)

    w_left = [1e7] * 3 + [2000] * 3
    w_right = [1e7] * 3 + [2000] * 3

    # w_dq_arm = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    # w_dq = [1.0] * 4 + w_dq_arm + w_dq_arm
    # w_

    client = init_client()
    client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=50,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1.0],
    )
    r1pro = load_r1pro(client)
    reset_joint_positions(client, r1pro, r1pro_init_joint_state)

    fk_left = ik.left_arm_chain.forward_kinematics(
        {
            n: r1pro_init_joint_state[n]
            for n in ik.left_arm_chain.get_joint_parameter_names()
        }
    )
    p_left = fk_left.translation().as_vector().toarray().flatten()
    r_left = fk_left.rotation().as_quat()

    fk_right = ik.right_arm_chain.forward_kinematics(
        {
            n: r1pro_init_joint_state[n]
            for n in ik.right_arm_chain.get_joint_parameter_names()
        }
    )
    p_right = fk_right.translation().as_vector().toarray().flatten()
    r_right = fk_right.rotation().as_quat()

    joint_info_map = get_joint_name_info_map(client, r1pro)
    joint_indices = [joint_info_map[n][0] for n in ik.joint_names]

    sphere_left = create_visual_sphere(client, 0.05, p_left, (0, 1, 0, 0.7))
    sphere_right = create_visual_sphere(client, 0.05, p_right, (0, 0, 1, 0.7))

    video = start_recording_video(client, "ik_controller_stable_test.mp4")

    unit_quat = (0, 0, 0, 1)

    def jac_left():
        q = get_q(client, r1pro, ik.left_arm_chain.get_joint_parameter_names())
        return ik.left_arm_chain.jacobian(q).toarray()

    qn = [r1pro_init_joint_state[n] for n in ik.joint_names]

    try:
        t = 0.0
        while True:

            p_left_now = p_left + goal_left.get_position(t)
            p_right_now = p_right + goal_right.get_position(t)

            client.resetBasePositionAndOrientation(sphere_left, p_left_now, unit_quat)
            client.resetBasePositionAndOrientation(sphere_right, p_right_now, unit_quat)

            q = get_q(client, r1pro, ik.joint_names)

            config = {
                "pG_left": p_left_now,
                "pG_right": p_right_now,
                "rG_left": r_left,
                "rG_right": r_right,
                "w_dq": 0.01,
                "w_qn": 1e5,
                "qn": qn,
                "w_p": 1e8,
                "w_r": 1e8,
                "w_gaze": 1e6,
                "q": q,
            }

            print(config)
            print("---")
            # input()

            ik.reset(config)
            if ik.solve():
                print("--ik solved--")
            else:
                print("oh no!")
                sys.exit(0)

            print("-----")
            dq = ik.get_solution()
            print(dq)
            # input()

            # dq_left = [None] * diffik.left_arm_chain.dof
            # for i, n in enumerate(diffik.left_arm_chain.get_joint_parameter_names()):
            #     dq_left[i] = dq[diffik.joint_names.index(n)]

            # v_act = jac_left() @ dq_left

            q += PYBULLET_TIME_STEP * dq

            client.setJointMotorControlArray(
                bodyUniqueId=r1pro,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=q.tolist(),
            )

            step(client)
            t += PYBULLET_TIME_STEP

            # if t > 1:
            #     break
    except KeyboardInterrupt:
        pass
    finally:

        stop_recording_video(client, video)

        client.disconnect()

        # import matplotlib.pyplot as plt

        # vel_goal_data = np.array(vel_goal_data)
        # vel_act_data = np.array(vel_act_data)

        # t = np.linspace(0, 1, len(vel_act_data))
        # fig, ax = plt.subplots(3, 1, sharex=True)

        # for i in range(3):
        #     ax[i].plot(t, vel_goal_data[:, i], "-", label="goal")
        #     ax[i].plot(t, vel_act_data[:, i], "-", label="act")
        #     ax[i].legend()

        # plt.show()

        # print(vel_goal_data.shape, "---------------")
        # fig.savefig("out.png")


if __name__ == "__main__":
    main()
