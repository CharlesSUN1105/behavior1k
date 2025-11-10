import numpy as np
from b1k.pybullet_utils import (
    p,
    init_client,
    load_r1pro,
    step,
    get_wheel_joint_infos,
    reset_joint_positions,
    start_recording_video,
    stop_recording_video,
    PYBULLET_TIME_STEP,
)
from b1k.models.r1pro.constants import r1pro_init_joint_state


def main():
    client = init_client()
    client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=[1, 0, 1.0],
    )
    r1pro = load_r1pro(client)
    reset_joint_positions(client, r1pro, r1pro_init_joint_state)

    target_velocities = np.deg2rad([120.0, 120.0, 120.0]).tolist()

    wheel_joint_infos = get_wheel_joint_infos(client, r1pro)

    client.setJointMotorControlArray(
        bodyUniqueId=r1pro,
        jointIndices=[info[0] for info in wheel_joint_infos],
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=target_velocities,
    )

    video = start_recording_video(client, "move_base.mp4")

    duration = 8.0
    t = 0.0
    while t <= duration:
        step(client)
        t += PYBULLET_TIME_STEP

    stop_recording_video(video)


if __name__ == "__main__":
    main()
