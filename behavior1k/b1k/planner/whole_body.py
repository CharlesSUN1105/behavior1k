import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from b1k.program import Program
from b1k.casadi_utils import vectorize, zeros_like, ones_like, INFTY
from b1k.collisions import (
    collision_links_with_base_link,
    collision_links,
    CollisionSphere,
)
from scipy.interpolate import interp1d


class WholeBodyPlanner(Program):
    """Note, 'whole body' means, torso, and left/right arms (no base navigation)."""

    def __init__(
        self,
        T: int,
        collision_sphere_names: list[str] = [],
        joint_limit_safety: float = 0.99,
    ):
        super().__init__()

        self.T = T  # knot points
        self.collision_sphere_names = collision_sphere_names

        left_arm_chain = self.chain.get_serial("left_gripper_link")
        right_arm_chain = self.chain.get_serial("right_gripper_link")

        self.joint_names = []
        for name in left_arm_chain.get_joint_parameter_names():
            if name not in self.joint_names:
                self.joint_names.append(name)
        for name in right_arm_chain.get_joint_parameter_names():
            if name not in self.joint_names:
                self.joint_names.append(name)

        # Decision variables
        dof = len(self.joint_names)
        self.dof = dof
        Q = cs.SX.sym("Q", dof, T)
        dQ = cs.SX.sym("dQ", dof, T)
        ddQ = cs.SX.sym("ddQ", dof, T)

        # Split up the Q
        q_from_name = lambda n: Q[self.joint_names.index(n), :]
        Q_left = cs.vertcat(
            *[q_from_name(n) for n in left_arm_chain.get_joint_parameter_names()]
        )
        Q_right = cs.vertcat(
            *[q_from_name(n) for n in right_arm_chain.get_joint_parameter_names()]
        )

        # Parameters
        duration = cs.SX.sym("duration")  # duration for the
        q0 = cs.SX.sym("q0", dof)  # initial joint state
        qF = cs.SX.sym("qF", dof)  # final joint state
        w_dQ = cs.SX.sym("q_dQ")  # joint velocity weight
        w_ddQ = cs.SX.sym("q_ddQ")  # joint acceleration weight
        collision_spheres = [
            CollisionSphere(
                n,
                cs.SX.sym(f"{n}_sphere_radius"),
                cs.SX.sym(f"{n}_sphere_position", 3),
            )
            for n in collision_sphere_names
        ]

        # Misc variables
        dt = duration / float(T - 1)  # time step
        g = []  # constraints
        lbg = []  # lower bound constraints
        ubg = []  # upper bound constraints

        # Cost function
        cost = 0.0
        cost += w_dQ * cs.sumsqr(dQ)
        cost += w_ddQ * cs.sumsqr(ddQ)

        # Forward Euler integration
        for t in range(T - 1):
            g.append(Q[:, t + 1] - (Q[:, t] + dt * dQ[:, t]))
            lbg.append(cs.DM.zeros(dof, 1))
            ubg.append(cs.DM.zeros(dof, 1))

            g.append(dQ[:, t + 1] - (dQ[:, t] + dt * ddQ[:, t]))
            lbg.append(cs.DM.zeros(dof, 1))
            ubg.append(cs.DM.zeros(dof, 1))

        # Initial position
        g.append(Q[:, 0] - q0)
        lbg.append(cs.DM.zeros(dof, 1))
        ubg.append(cs.DM.zeros(dof, 1))

        # Final position
        g.append(Q[:, -1] - qF)
        lbg.append(cs.DM.zeros(dof, 1))
        ubg.append(cs.DM.zeros(dof, 1))

        # Initial velcoity = 0
        g.append(dQ[:, 0])
        lbg.append(zeros_like(dQ[:, 0]))
        ubg.append(zeros_like(dQ[:, 0]))

        # Final velocity = 0
        g.append(dQ[:, -1])
        lbg.append(zeros_like(dQ[:, -1]))
        ubg.append(zeros_like(dQ[:, -1]))

        # Initial acceleration = 0
        g.append(ddQ[:, 0])
        lbg.append(zeros_like(ddQ[:, 0]))
        ubg.append(zeros_like(ddQ[:, 0]))

        # Final acceleration = 0
        g.append(ddQ[:, -1])
        lbg.append(zeros_like(ddQ[:, -1]))
        ubg.append(zeros_like(ddQ[:, -1]))

        # Joint limits
        joint_limits = {}
        left_arm_limits = left_arm_chain.get_joint_limits()
        right_arm_limits = right_arm_chain.get_joint_limits()
        for j, n in enumerate(left_arm_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = left_arm_limits[j]
        for j, n in enumerate(right_arm_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = right_arm_limits[j]

        for j, n in enumerate(self.joint_names):
            g.append(Q[j, :])
            lbg.append(joint_limit_safety * joint_limits[n].lower * ones_like(Q[j, :]))
            ubg.append(joint_limit_safety * joint_limits[n].upper * ones_like(Q[j, :]))

            g.append(dQ[j, :])
            lbg.append(-joint_limits[n].velocity * ones_like(dQ[j, :]))
            ubg.append(joint_limits[n].velocity * ones_like(dQ[j, :]))

        # Collision avoidance with collision spheres (assumes start/end are collision-free)
        for t in range(T - 1):
            # iter over time steps

            q = {n: Q[j, t] for j, n in enumerate(self.joint_names)}

            Tf = self.chain.forward_kinematics(q)  # transforms for all links in robot

            for link_name, tf in Tf.items():
                # iter over each link in Tf

                for cl in collision_links_with_base_link(link_name, collision_links):
                    # iter over each col link

                    rad = cl.radius  # radius of collision link
                    pos = tf.translation().as_vector()  # position of collision link
                    if cl.offset:
                        pos += tf.rotation().as_matrix() @ cs.DM(cl.offset)

                    for colsphere in collision_spheres:
                        # iter over each collision sphere

                        d2 = cs.sumsqr(pos - colsphere.position)
                        r2 = (colsphere.radius + rad) ** 2
                        g.append(d2 - r2)
                        lbg.append(0.0)
                        ubg.append(INFTY)

        # Vectorize decision variables and parameters and constraints
        x = cs.vertcat(cs.vec(Q), cs.vec(dQ), cs.vec(ddQ))
        p = cs.vertcat(
            duration,
            q0,
            qF,
            w_dQ,
            w_ddQ,
        )
        for colsphere in collision_spheres:
            p = cs.vertcat(p, colsphere.radius, colsphere.position)
        g = vectorize(g)

        # Setup NLP
        prob = {"x": x, "p": p, "f": cost, "g": g}
        self.solver = cs.nlpsol("solver", "ipopt", prob)
        self.lbg = vectorize(lbg)
        self.ubg = vectorize(ubg)

    def reset(self, config: dict):
        self.config = config
        x0 = [
            config["Q0"],
            config["dQ0"],
            config["ddQ0"],
        ]
        p = [
            config["duration"],
            config["q0"],
            config["qF"],
            config["w_dQ"],
            config["w_ddQ"],
        ]
        collision_spheres = config.get("collision_spheres", [])
        if not collision_spheres and self.collision_sphere_names:
            raise ValueError(
                "you need to provide the collision spheres in the config for reset()"
            )

        def collision_sphere_from_list(n):
            for colsphere in collision_spheres:
                if colsphere.name == n:
                    return colsphere
            else:
                raise ValueError(f"did not find {n}")

        for n in self.collision_sphere_names:
            colsphere = collision_sphere_from_list(n)
            p.append(colsphere.radius)
            p.append(colsphere.position)

        self.x0 = vectorize(x0)
        self.p = vectorize(p)

    def solve(self) -> bool:
        self.solution = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        return stats["success"]

    def get_solution(self) -> dict:
        dof = self.dof
        x = self.solution["x"].toarray().flatten()

        # Note, order=F since we used cs.vec to create decision variable vector x
        Q_, dQ_, ddQ_ = np.split(x, 3)
        Q = Q_.reshape(dof, self.T, order="F")
        dQ = dQ_.reshape(dof, self.T, order="F")
        ddQ = ddQ_.reshape(dof, self.T, order="F")
        # Q = x[: dof * self.T].reshape(dof, self.T, order="F")

        t = np.linspace(0.0, self.config["duration"], self.T)
        Qsol = {
            n: interp1d(t, Q[j, :], kind="cubic")
            for j, n in enumerate(self.joint_names)
        }

        dQsol = {
            n: interp1d(t, dQ[j, :], kind="cubic")
            for j, n in enumerate(self.joint_names)
        }
        ddQsol = {
            n: interp1d(t, ddQ[j, :], kind="cubic")
            for j, n in enumerate(self.joint_names)
        }

        return Qsol, dQsol, ddQsol

    def plot_solution(self):
        qsol, dqsol, ddqsol = self.get_solution()
        fig, ax = plt.subplots(len(qsol), 1, sharex=True, figsize=(10, 30))
        t = np.linspace(0.0, self.config["duration"])

        for i, (n, q) in enumerate(qsol.items()):
            ax[i].plot(t, q(t))
            ax[i].set_ylabel(n)
            ax[i].grid()

        ax[-1].set_xlabel("Time")

        fig.savefig("qsol.png")
        print("Saved", "qsol.png")
