import sys
import numpy as np
from math import radians
from random import uniform
from b1k.kinematics.chain import Chain
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
    load_r1pro_visual,
    PYBULLET_TIME_STEP,
    start_recording_video,
    stop_recording_video,
)
from b1k.collisions import (
    collision_links,
    collision_links_with_base_link,
    CollisionSphere,
)
from b1k.models.r1pro.constants import r1pro_init_joint_state
from b1k.planner.whole_body import WholeBodyPlanner


def sample_collision_spheres(chain: Chain, q0, qf, nspheres: int, max_attempts=10_000):

    T0 = chain.forward_kinematics(q0)
    Tf = chain.forward_kinematics(qf)
    done = False
    left = np.array([-0.6, -1, 0.7]).reshape(3, 1)
    right = np.array([0.6, 1, 1.3]).reshape(3, 1)
    attempt = 0
    while not done:

        radius = np.random.uniform(0.05, 0.15, size=(nspheres,))
        positions = np.random.uniform(left, right, size=(3, nspheres))

        collision_free = []
        for link_name, tf in T0.items():
            for cl in collision_links_with_base_link(link_name, collision_links):
                pos = tf.translation().as_vector().toarray().flatten()
                if cl.offset:
                    pos += tf.rotation().as_matrix().toarray() @ np.asarray(cl.offset)

                for i in range(nspheres):
                    d = np.linalg.norm(pos.flatten() - positions[:, i].flatten())
                    r = cl.radius + radius[i]
                    collision_free.append(d >= r)

        for link_name, tf in Tf.items():
            for cl in collision_links_with_base_link(link_name, collision_links):
                pos = tf.translation().as_vector().toarray().flatten()
                if cl.offset:
                    pos += tf.rotation().as_matrix().toarray() @ np.asarray(cl.offset)

                for i in range(nspheres):
                    d = np.linalg.norm(pos.flatten() - positions[:, i].flatten())
                    r = cl.radius + radius[i]
                    collision_free.append(d >= r)

        done = all(collision_free)

        attempt += 1
        if attempt > max_attempts and not done:
            raise RuntimeError("max attempts reached")

    return radius, positions


def main():

    T = 100
    T0 = 30
    nspheres = 20
    collision_sphere_names = [f"sphere{i}" for i in range(nspheres)]
    planner = WholeBodyPlanner(T, collision_sphere_names)

    # define a planner without sphere collisions and less knot points
    planner0 = WholeBodyPlanner(T0)

    q0 = [r1pro_init_joint_state[n] for n in planner.joint_names]

    q_rand = planner.chain.sample_joint_position()
    qF = []
    for n in planner.joint_names:
        if n.startswith("torso"):
            qF.append(r1pro_init_joint_state[n] + radians(uniform(-20, 20)))
        else:
            qF.append(q_rand[n])

    qF_dict = {n: qF[j] for j, n in enumerate(planner.joint_names)}

    sphere_radius, sphere_positions = sample_collision_spheres(
        planner.chain, r1pro_init_joint_state, qF_dict, nspheres
    )

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
    reset_joint_positions(client, r1pro_visual, qF_dict)

    red = (1.0, 0.0, 0.0, 0.9)
    for i in range(nspheres):
        create_sphere(
            # create_visual_sphere(
            client,
            sphere_radius[i],
            sphere_positions[:, i].flatten(),
            red,
        )

    duration = 10.0

    time = np.linspace(0, duration, T0)
    Q0 = np.linspace(q0, qF, T0)
    dQ0 = np.gradient(Q0, time, axis=0)
    ddQ0 = np.gradient(dQ0, time, axis=0)

    config0 = {
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
        "w_dQ": 100.0,
        "w_ddQ": 0.01,
    }

    planner0.reset(config0)
    planner0.solve()
    qsol0, dqsol0, ddqsol0 = planner0.get_solution()

    time = np.linspace(0, duration, T)
    # Q0 = np.linspace(q0, qF, T)
    # dQ0 = np.gradient(Q0, time, axis=0)
    # ddQ0 = np.gradient(dQ0, time, axis=0)

    Q0 = np.stack([qsol0[n](time) for n in planner.joint_names])
    dQ0 = np.stack([dqsol0[n](time) for n in planner.joint_names])
    ddQ0 = np.stack([ddqsol0[n](time) for n in planner.joint_names])

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
        "w_dQ": 100.0,
        "w_ddQ": 0.01,
        "collision_spheres": [
            CollisionSphere(
                n, float(sphere_radius[i]), sphere_positions[:, i].flatten().tolist()
            )
            for i, n in enumerate(collision_sphere_names)
        ],
    }

    planner.reset(config)
    success = planner.solve()
    if success:
        print("solver succeeded!")
    else:
        print("solver failed")
        sys.exit(0)
    qsol, _, _ = planner.get_solution()
    planner.plot_solution()

    print(">" * 50)
    print("Expected qF")
    print(qF_dict)

    print("Solution qF")
    print({n: qsol[n](duration) for n in planner.joint_names})
    print("<" * 50)

    # Get joint indices
    joint_info_map = get_joint_name_info_map(client, r1pro)
    joint_indices = [joint_info_map[n][0] for n in planner.joint_names]

    # Control joints not used by planner with zero velocity to prevent robot breaking
    for n, info in joint_info_map.items():
        if n not in planner.joint_names:
            client.setJointMotorControl2(
                r1pro,
                jointIndex=joint_info_map[n][0],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
            )

    # Execute plan
    video = start_recording_video(client, "move_whole_body.mp4")
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
            else:
                client.setJointMotorControlArray(
                    bodyUniqueId=r1pro,
                    jointIndices=joint_indices,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[qsol[n](duration) for n in planner.joint_names],
                )

            step(client)
            t += PYBULLET_TIME_STEP
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording_video(client, video)


if __name__ == "__main__":
    main()
