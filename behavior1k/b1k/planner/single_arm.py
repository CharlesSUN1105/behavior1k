import numpy as np
import casadi as cs
from b1k.program import Program
from b1k.casadi_utils import vectorize, zeros_like, ones_like, INFTY
from scipy.interpolate import interp1d


class SingleArmAndTorsoPlanner(Program):

    def __init__(self, side: str, T: int, joint_limit_safety: float = 0.99):
        super().__init__()

        # Setup
        assert side in {"left", "right"}, f"'{side}' not recognized"
        self.side = side
        self.T = T  # knot points
        self.end_effector_link_name = f"{side}_gripper_link"
        self.serial_chain = self.chain.get_serial(self.end_effector_link_name)
        self.joint_names = self.serial_chain.get_joint_parameter_names()
        dof = self.serial_chain.dof

        # Joint state variables
        Q = cs.SX.sym("Q", dof, T)  # position
        dQ = cs.SX.sym("dQ", dof, T)  # velocity
        ddQ = cs.SX.sym("ddQ", dof, T)  # acceleration

        # Parameters
        duration = cs.SX.sym("duration")  # duration for the
        q0 = cs.SX.sym("q0", dof)  # initial joint state
        qF = cs.SX.sym("qF", dof)  # final joint state
        w_dQ = cs.SX.sym("w_dQ")  # joint velocity weight
        w_ddQ = cs.SX.sym("w_ddQ")  # joint acceleration weight

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
        joint_limits = self.serial_chain.get_joint_limits()
        for joint_index, joint_name in enumerate(self.joint_names):
            limit = joint_limits[joint_index]

            g.append(Q[joint_index, :])
            lbg.append(joint_limit_safety * limit.lower * ones_like(Q[joint_index, :]))
            ubg.append(joint_limit_safety * limit.upper * ones_like(Q[joint_index, :]))

            g.append(dQ[joint_index, :])
            lbg.append(-limit.velocity * ones_like(dQ[joint_index, :]))
            ubg.append(limit.velocity * ones_like(dQ[joint_index, :]))

        # Vectorize decision variables and parameters and constraints
        x = cs.vertcat(cs.vec(Q), cs.vec(dQ), cs.vec(ddQ))
        p = cs.vertcat(
            duration,
            q0,
            qF,
            w_dQ,
            w_ddQ,
        )
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
        self.x0 = vectorize(x0)
        self.p = vectorize(p)

    def solve(self) -> bool:
        self.solution = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        return stats["success"]

    def get_solution(self) -> dict:
        dof = self.serial_chain.dof
        x = self.solution["x"].toarray().flatten()

        # Note, order=F since we used cs.vec to create decision variable vector x
        Q = x[: dof * self.T].reshape(dof, self.T, order="F")

        t = np.linspace(0.0, self.config["duration"], self.T)
        return {
            n: interp1d(t, Q[j, :], kind="cubic")
            for j, n in enumerate(self.joint_names)
        }


class SingleArmAndTorsoConstraintPlanner(Program):

    def __init__(self, side: str, T: int, joint_limit_safety: float = 0.99):
        super().__init__()

        # Setup
        assert side in {"left", "right"}, f"'{side}' not recognized"
        self.side = side
        self.T = T  # knot points
        self.end_effector_link_name = f"{side}_gripper_link"
        self.serial_chain = self.chain.get_serial(self.end_effector_link_name)
        self.zed_chain = self.chain.get_serial("zed_link")
        self.joint_names = self.serial_chain.get_joint_parameter_names()
        dof = self.serial_chain.dof

        # Joint state variables
        Q = cs.SX.sym("Q", dof, T)  # position
        dQ = cs.SX.sym("dQ", dof, T)  # velocity
        ddQ = cs.SX.sym("ddQ", dof, T)  # acceleration

        Q_zed = cs.vertcat(
            *[
                Q[self.joint_names.index(n), :]
                for n in self.zed_chain.get_joint_parameter_names()
            ]
        )

        # Parameters
        duration = cs.SX.sym("duration")  # duration for the
        q0 = cs.SX.sym("q0", dof)  # initial joint state
        dr = cs.SX.sym("dr", 3)  # direction vector wrt robot base
        d = cs.SX.sym("d")  # distance to move end-effector
        w_dQ = cs.SX.sym("w_dQ")  # joint velocity weight
        w_ddQ = cs.SX.sym("w_ddQ")  # joint acceleration weight
        p_err_max = cs.SX.sym("p_err_max")  # maximum displacmenet from constraint
        w_r = cs.SX.sym("w_r")  # eff rotation weight

        # gaze at pG, 0->off, positive->on (used as weight in cost term)
        maintain_gaze = cs.SX.sym("maintain_gaze")

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
        joint_limits = self.serial_chain.get_joint_limits()
        for joint_index, joint_name in enumerate(self.joint_names):
            limit = joint_limits[joint_index]

            g.append(Q[joint_index, :])
            lbg.append(joint_limit_safety * limit.lower * ones_like(Q[joint_index, :]))
            ubg.append(joint_limit_safety * limit.upper * ones_like(Q[joint_index, :]))

            g.append(dQ[joint_index, :])
            lbg.append(-limit.velocity * ones_like(dQ[joint_index, :]))
            ubg.append(limit.velocity * ones_like(dQ[joint_index, :]))

        # Line constraint + gaze constraint
        dr_nrm = dr / cs.norm_fro(dr)  # ensure normalized

        tf0 = self.serial_chain.forward_kinematics(q0)
        p0 = tf0.translation().as_vector()
        r0 = tf0.rotation()
        alpha = np.linspace(0, 1, T)
        for t in range(T):
            tf = self.serial_chain.forward_kinematics(Q[:, t])
            p = tf.translation().as_vector()
            l = p0 + alpha[t] * d * dr_nrm

            # Line constraint
            g.append(p_err_max - cs.sumsqr(p - l))
            lbg.append(0.0)
            ubg.append(INFTY)

            # Gaze constraint
            tf_zed = self.zed_chain.forward_kinematics(Q_zed[:, t])
            e_zed = tf_zed.translation().as_vector()
            z_zed = tf_zed.rotation().as_matrix()[:, 2]

            alpha_star = cs.dot(l - e_zed, z_zed)
            gaze_dist2 = cs.sumsqr(e_zed + alpha_star * z_zed - l)

            cost += maintain_gaze * gaze_dist2

            # Eff orientation (minimize eff rotation velocity)
            J = self.serial_chain.jacobian(Q[:, t])
            cost += w_r * cs.sumsqr(J @ dQ[:, t])

        # Vectorize decision variables and parameters and constraints
        x = cs.vertcat(cs.vec(Q), cs.vec(dQ), cs.vec(ddQ))
        p = cs.vertcat(
            duration,
            q0,
            dr,
            d,
            w_dQ,
            w_ddQ,
            w_r,
            p_err_max,
            maintain_gaze,
        )
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
            config["dr"],
            config["d"],
            config["w_dQ"],
            config["w_ddQ"],
            config["w_r"],
            config["p_err_max"],
            config["maintain_gaze"],
        ]
        self.x0 = vectorize(x0)
        self.p = vectorize(p)

    def solve(self) -> bool:
        self.solution = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        return stats["success"]

    def get_solution(self) -> dict:
        dof = self.serial_chain.dof
        x = self.solution["x"].toarray().flatten()

        # Note, order=F since we used cs.vec to create decision variable vector x
        Q = x[: dof * self.T].reshape(dof, self.T, order="F")

        t = np.linspace(0.0, self.config["duration"], self.T)
        return {
            n: interp1d(t, Q[j, :], kind="cubic")
            for j, n in enumerate(self.joint_names)
        }
