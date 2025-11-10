from b1k.pybullet_utils import (
    init_client,
    load_r1pro,
    reset_joint_positions,
    create_visual_sphere,
    start_recording_video,
    stop_recording_video,
)
from math import radians
from b1k.models.r1pro.constants import r1pro_T_joint_state, r1pro_init_joint_state
from b1k.kinematics import build_r1pro_chain
from b1k.collisions import (
    collision_links_with_base_link,
    collision_links,
)
import numpy as np
import time


def main():
    client = init_client()
    r1pro = load_r1pro(client)

    # Switch between following to visualize r1pro in different states
    joint_state = r1pro_init_joint_state
    # joint_state = r1pro_T_joint_state

    reset_joint_positions(client, r1pro, joint_state)

    chain = build_r1pro_chain()
    T = chain.forward_kinematics(joint_state)

    for link_name, tf in T.items():
        # print(link_name)

        for cl in collision_links_with_base_link(link_name, collision_links):
            rad = cl.radius
            color = cl.color

            pos = tf.translation().as_vector().toarray().flatten()
            rot = tf.rotation().as_matrix().toarray()
            if cl.offset:
                pos += rot @ np.asarray(cl.offset)
            pos = pos.tolist()

            create_visual_sphere(client, rad, pos, color)

    video = start_recording_video(client, "visualize_collision_spheres.mp4")
    angle = 0.0
    angle_step = radians(20)
    try:
        while True:
            client.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=angle,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 1.0],
            )
            angle += angle_step
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording_video(video)


if __name__ == "__main__":
    main()
