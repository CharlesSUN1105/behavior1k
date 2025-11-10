import casadi as cs
import numpy as np
from b1k.program import Program
from b1k.casadi_utils import vectorize, zeros_like, ones_like, INFTY


class StableBodyTestDifferentialIK(Program):

    def __init__(
        self,
        dt,
        joint_limit_safety: float = 0.99,
    ):
        super().__init__()

        # Setup
        self.left_arm_chain = self.chain.get_serial("left_gripper_link")
        self.right_arm_chain = self.chain.get_serial("right_gripper_link")

        self.joint_names = []
        for name in self.left_arm_chain.get_joint_parameter_names():
            if name not in self.joint_names:
                self.joint_names.append(name)
        for name in self.right_arm_chain.get_joint_parameter_names():
            if name not in self.joint_names:
                self.joint_names.append(name)

        self.dof = len(self.joint_names)
        cost = 0.0
        g = []  # constraints
        lbg = []  # lower bound constraints
        ubg = []  # upper bound constraints

        # Decision variables
        dq = cs.SX.sym("dq", self.dof)

        # Parameters
        pG_left = cs.SX.sym("pG_left", 3)
        pG_right = cs.SX.sym("pG_right", 3)
        q = cs.SX.sym("q", self.dof)  # current joint positions
        stable = cs.SX.sym("stable")  # 1->on, 0->off

        # Misc variables
        q_from_name = lambda n: q[self.joint_names.index(n)]
        q_left = cs.vertcat(
            *[q_from_name(n) for n in self.left_arm_chain.get_joint_parameter_names()]
        )
        q_right = cs.vertcat(
            *[q_from_name(n) for n in self.right_arm_chain.get_joint_parameter_names()]
        )

        dq_from_name = lambda n: dq[self.joint_names.index(n)]
        dq_left = cs.vertcat(
            *[dq_from_name(n) for n in self.left_arm_chain.get_joint_parameter_names()]
        )
        dq_right = cs.vertcat(
            *[dq_from_name(n) for n in self.right_arm_chain.get_joint_parameter_names()]
        )

        qnext = q + dt * dq
        qnext_left = q_left + dt * dq_left
        qnext_right = q_right + dt * dq_right

        # Cost: reach to target
        fk_left = self.left_arm_chain.forward_kinematics(qnext_left)
        p_left = fk_left.translation().as_vector()

        fk_right = self.right_arm_chain.forward_kinematics(qnext_right)
        p_right = fk_right.translation().as_vector()

        cost += 1e6 * cs.sumsqr(p_left - pG_left)
        cost += 1e6 * cs.sumsqr(p_right - pG_right)

        # Cost: minimize joint velocity
        # cost += 1e-4 * cs.sumsqr(dq)

        # Constraint: stable robot
        lbg_, g_, ubg_ = self.stable_robot_constraint(
            {n: qnext[j] for j, n in enumerate(self.joint_names)}
        )

        g.append(stable * g_)
        lbg.append(lbg_)
        ubg.append(ubg_)

        # Joint limits
        joint_limits = {}
        left_arm_limits = self.left_arm_chain.get_joint_limits()
        right_arm_limits = self.right_arm_chain.get_joint_limits()
        for j, n in enumerate(self.left_arm_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = left_arm_limits[j]
        for j, n in enumerate(self.right_arm_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = right_arm_limits[j]

        for j, n in enumerate(self.joint_names):
            g.append(qnext[j])
            lbg.append(joint_limit_safety * joint_limits[n].lower)
            ubg.append(joint_limit_safety * joint_limits[n].upper)

            # g.append(dq[j])
            # lbg.append(-joint_limits[n].velocity)
            # ubg.append(joint_limits[n].velocity)

        # Vectorize decision variables and parameters and constraints
        x = dq
        p = cs.vertcat(*[pG_left, pG_right, q, stable])
        g = vectorize(g)

        # Setup NLP
        prob = {"x": x, "p": p, "f": cost, "g": g}
        self.solver = cs.nlpsol("solver", "ipopt", prob)
        self.lbg = vectorize(lbg)
        self.ubg = vectorize(ubg)

    def reset(self, config: dict):
        self.config = config

        if self.solution is not None:
            # use previous solution as warm start
            self.x0 = self.solution
        else:
            self.x0 = cs.vec(np.zeros(self.dof))

        p = [
            config["pG_left"],
            config["pG_right"],
            config["q"],
            config["stable"],
        ]

        self.p = vectorize(p)

    def solve(self) -> bool:
        solution = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        self.solution = solution["x"]
        stats = self.solver.stats()
        return stats["success"]

    def get_solution(self) -> dict:
        dq = self.solution.toarray().flatten()
        # return {n: dq[j] for j, n in enumerate(self.joint_names)}
        return dq  # maybe more useful as array


class WholeBodyDifferentialIK(Program):

    def __init__(
        self,
        dt: float,  # controller time step
        joint_limit_safety: float = 0.99,
    ):
        super().__init__()

        # Setup
        self.left_arm_chain = self.chain.get_serial("left_gripper_link")
        self.right_arm_chain = self.chain.get_serial("right_gripper_link")

        self.joint_names = []
        for name in self.left_arm_chain.get_joint_parameter_names():
            if name not in self.joint_names:
                self.joint_names.append(name)
        for name in self.right_arm_chain.get_joint_parameter_names():
            if name not in self.joint_names:
                self.joint_names.append(name)

        self.dof = len(self.joint_names)
        cost = 0.0
        g = []  # constraints
        lbg = []  # lower bound constraints
        ubg = []  # upper bound constraints

        # Decision variables
        dq = cs.SX.sym("dq", self.dof)

        # Parameters
        vG_left = cs.SX.sym("vG_left", 6)  # twist goal for left hand
        vG_right = cs.SX.sym("vG_right", 6)  # twist goal for right hand
        w_left = cs.SX.sym("w_left", 6)  # weight vector for left goal cost term
        w_right = cs.SX.sym("w_right", 6)  # weight vector for right goal cost term
        w_dq = cs.SX.sym("w_dq", self.dof)  # weight on the joint velocities
        q = cs.SX.sym("q", self.dof)  # current joint positions

        q_from_name = lambda n: q[self.joint_names.index(n)]
        q_left = cs.vertcat(
            *[q_from_name(n) for n in self.left_arm_chain.get_joint_parameter_names()]
        )
        q_right = cs.vertcat(
            *[q_from_name(n) for n in self.right_arm_chain.get_joint_parameter_names()]
        )

        dq_from_name = lambda n: dq[self.joint_names.index(n)]
        dq_left = cs.vertcat(
            *[dq_from_name(n) for n in self.left_arm_chain.get_joint_parameter_names()]
        )
        dq_right = cs.vertcat(
            *[dq_from_name(n) for n in self.right_arm_chain.get_joint_parameter_names()]
        )

        # Goal end-effector velocities
        J_left = self.left_arm_chain.jacobian(q_left)
        diff_left = J_left @ dq_left - vG_left
        cost += diff_left.T @ cs.diag(w_left) @ diff_left
        # cost += cs.sumsqr(diff_left) * 100

        J_right = self.right_arm_chain.jacobian(q_right)
        diff_right = J_right @ dq_right - vG_right
        cost += diff_right.T @ cs.diag(w_right) @ diff_right
        # cost += cs.sumsqr(diff_right) * 100

        # Minimize joint velocities
        # cost += dq.T @ cs.diag(w_dq) @ dq
        # cost += 0.01 * cs.sumsqr(dq)

        # Estimate next joint state
        qnext = q + dt * dq

        # Joint limits
        joint_limits = {}
        left_arm_limits = self.left_arm_chain.get_joint_limits()
        right_arm_limits = self.right_arm_chain.get_joint_limits()
        for j, n in enumerate(self.left_arm_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = left_arm_limits[j]
        for j, n in enumerate(self.right_arm_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = right_arm_limits[j]

        for j, n in enumerate(self.joint_names):
            g.append(qnext[j])
            lbg.append(joint_limit_safety * joint_limits[n].lower)
            ubg.append(joint_limit_safety * joint_limits[n].upper)

            g.append(dq[j])
            lbg.append(-joint_limits[n].velocity)
            ubg.append(joint_limits[n].velocity)

        # Vectorize decision variables and parameters and constraints
        x = dq
        p = cs.vertcat(
            *[
                vG_left,
                vG_right,
                w_left,
                w_right,
                w_dq,
                q,
            ]
        )
        g = vectorize(g)

        # Setup NLP
        prob = {"x": x, "p": p, "f": cost, "g": g}
        # self.solver = cs.nlpsol("solver", "ipopt", prob)
        self.solver = cs.qpsol("solver", "osqp", prob)
        self.lbg = vectorize(lbg)
        self.ubg = vectorize(ubg)

    def reset(self, config: dict):
        self.config = config

        if self.solution is not None:
            # use previous solution as warm start
            self.x0 = self.solution
        else:
            self.x0 = cs.vec(np.zeros(self.dof))

        p = [
            config["vG_left"],
            config["vG_right"],
            config["w_left"],
            config["w_right"],
            config["w_dq"],
            config["q"],
        ]

        self.p = vectorize(p)

    def solve(self) -> bool:
        solution = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        self.solution = solution["x"]
        stats = self.solver.stats()
        return stats["success"]

    def get_solution(self) -> dict:
        dq = self.solution.toarray().flatten()
        # return {n: dq[j] for j, n in enumerate(self.joint_names)}
        return dq  # maybe more useful as array
