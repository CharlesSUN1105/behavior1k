import casadi as cs
import numpy as np
import spatial_casadi as sc
from b1k.kinematics.chain import Chain
from b1k.casadi_utils import INFTY, vectorize
from b1k.program import Program
from b1k.collisions import (
    collision_links_with_base_link,
    collision_links,
    CollisionSphere,
)

"""'Note, 'Global' refers to the fact that the IK solver can find a
solution anywhere in the configuration space. It should not be used as
a task-space controller. This is because even small changes in task
space can mean a large change in configuration space due to
singularities."""


class GlobalSingleArmIK(Program):

    def __init__(self, side: str, joint_limit_safety: float = 0.99):
        super().__init__()

        # Setup
        assert side in {"left", "right"}, f"'{side}' not recognized"
        self.side = side
        self.end_effector_link_name = f"{side}_gripper_link"
        self.serial_chain = self.chain.get_serial(self.end_effector_link_name)
        self.zed_chain = self.chain.get_serial("zed_link")
        self.joint_names = self.serial_chain.get_joint_parameter_names()
        dof = self.serial_chain.dof

        # Decision variables
        q = cs.SX.sym("q", dof)

        q_zed = cs.vertcat(
            *[
                q[self.joint_names.index(n)]
                for n in self.zed_chain.get_joint_parameter_names()
            ]
        )

        # Parameters
        pG = cs.SX.sym("pG", 3)  # position goal, in robot base frame
        rG = cs.SX.sym("rG", 4)  # rotation goal, in robot base frame
        qn = cs.SX.sym("qn", dof)  # nominal configuration
        p_err_max = cs.SX.sym("p_err_max")  # maximum error on the position
        r_err_max = cs.SX.sym("r_err_max")  # maximum error on the rotation

        # gaze at pG, 0->off, positive->on (used as weight in cost term)
        maintain_gaze = cs.SX.sym("maintain_gaze")

        # Misc variables
        g = []  # constraints
        lbg = []  # lower bound constraints
        ubg = []  # upper bound constraints

        # Cost: keep close to nominal
        cost = cs.sumsqr(q - qn)

        # Cost: maintain gaze (optional)
        Tf_zed = self.zed_chain.forward_kinematics(q_zed)
        e_zed = Tf_zed.translation().as_vector()
        z_zed = Tf_zed.rotation().as_matrix()[:, 2]
        alpha_star = cs.dot(pG - e_zed, z_zed)
        gaze_dist2 = cs.sumsqr(e_zed + alpha_star * z_zed - pG)
        cost += maintain_gaze * gaze_dist2

        # Final configuration
        tf = self.serial_chain.forward_kinematics(q)

        p = tf.translation().as_vector()
        g.append(p_err_max - cs.sumsqr(p - pG))
        lbg.append(0.0)
        ubg.append(INFTY)

        r = tf.rotation()
        g.append(r_err_max - (sc.Rotation.from_quat(rG) * r.inv()).magnitude() ** 2)
        lbg.append(0.0)
        ubg.append(INFTY)

        # Joint limits
        joint_limits = self.serial_chain.get_joint_limits()
        for joint_index, joint_name in enumerate(self.joint_names):
            limit = joint_limits[joint_index]

            g.append(q[joint_index])
            lbg.append(limit.lower * joint_limit_safety)
            ubg.append(limit.upper * joint_limit_safety)

        # Vectorize decision variables and parameters and constraints
        x = q
        p = cs.vertcat(pG, rG, qn, p_err_max, r_err_max, maintain_gaze)
        g = vectorize(g)

        # Setup NLP
        prob = {"x": x, "p": p, "f": cost, "g": g}
        self.solver = cs.nlpsol("solver", "ipopt", prob)
        self.lbg = vectorize(lbg)
        self.ubg = vectorize(ubg)

    def reset(self, config):
        self.config = config
        x0 = [config["q0"]]
        p = [
            config["pG"],
            config["rG"],
            config["q0"],  # use qn = q0
            config["p_err_max"],
            config["r_err_max"],
            config["maintain_gaze"],
        ]
        self.x0 = vectorize(x0)
        self.p = vectorize(p)

    def solve(self):
        self.solution = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        return stats["success"]

    def get_solution(self):
        qsol = self.solution["x"].toarray().flatten().tolist()
        return {n: q for n, q in zip(self.joint_names, qsol)}


class GlobalDualArmIK(Program):

    def __init__(
        self, joint_limit_safety: float = 0.99, collision_sphere_names: list[str] = []
    ):
        super().__init__()

        # Setup
        self.end_effector_link_name = lambda side: f"{side}_gripper_link"
        self.collision_sphere_names = collision_sphere_names
        self.left_eff_link_name = "left_gripper_link"
        self.right_eff_link_name = "right_gripper_link"
        self.left_serial_chain = self.chain.get_serial(self.left_eff_link_name)
        self.right_serial_chain = self.chain.get_serial(self.right_eff_link_name)
        self.zed_chain = self.chain.get_serial("zed_link")

        self.joint_names = []
        for n in self.left_serial_chain.get_joint_parameter_names():
            if n not in self.joint_names:
                self.joint_names.append(n)

        for n in self.right_serial_chain.get_joint_parameter_names():
            if n not in self.joint_names:
                self.joint_names.append(n)

        for n in self.zed_chain.get_joint_parameter_names():
            if n not in self.joint_names:
                self.joint_names.append(n)

        dof = len(self.joint_names)
        self.dof = dof

        # Decision variables
        q = cs.SX.sym("q", dof)

        q_left = cs.vertcat(
            *[
                q[self.joint_names.index(n)]
                for n in self.left_serial_chain.get_joint_parameter_names()
            ]
        )

        q_right = cs.vertcat(
            *[
                q[self.joint_names.index(n)]
                for n in self.right_serial_chain.get_joint_parameter_names()
            ]
        )

        q_zed = cs.vertcat(
            *[
                q[self.joint_names.index(n)]
                for n in self.zed_chain.get_joint_parameter_names()
            ]
        )

        # Parameters
        pG_left = cs.SX.sym("pG_left", 3)  # position goal for left, in robot base frame
        rG_left = cs.SX.sym("rG_left", 4)  # rotation goal for left, in robot base frame

        # position goal for right, in robot base frame
        pG_right = cs.SX.sym("pG_right", 3)

        # rotation goal for right, in robot base frame
        rG_right = cs.SX.sym("rG_right", 4)

        qn = cs.SX.sym("qn", dof)  # nominal configuration
        p_err_max = cs.SX.sym("p_err_max")  # maximum error on the position
        r_err_max = cs.SX.sym("r_err_max")  # maximum error on the rotation

        # gaze at pG, 0->off, positive->on (used as weight in cost term)
        maintain_gaze = cs.SX.sym("maintain_gaze")

        collision_spheres = [
            CollisionSphere(
                n,
                cs.SX.sym(f"{n}_sphere_radius"),
                cs.SX.sym(f"{n}_sphere_position", 3),
            )
            for n in collision_sphere_names
        ]

        # Misc variables
        g = []  # constraints
        lbg = []  # lower bound constraints
        ubg = []  # upper bound constraints

        # Cost: keep close to nominal
        cost = cs.sumsqr(q - qn)

        # Cost: maintain gaze (optional)
        Tf_zed = self.zed_chain.forward_kinematics(q_zed)
        gaze_goal = 0.5 * (pG_left + pG_right)
        e_zed = Tf_zed.translation().as_vector()
        z_zed = Tf_zed.rotation().as_matrix()[:, 2]
        alpha_star = cs.dot(gaze_goal - e_zed, z_zed)
        gaze_dist2 = cs.sumsqr(e_zed + alpha_star * z_zed - gaze_goal)
        cost += maintain_gaze * gaze_dist2

        # Final configuration
        tf_left = self.left_serial_chain.forward_kinematics(q_left)

        p_left = tf_left.translation().as_vector()
        g.append(p_err_max - cs.sumsqr(p_left - pG_left))
        lbg.append(0.0)
        ubg.append(INFTY)

        r_left = tf_left.rotation()
        g.append(
            r_err_max - (sc.Rotation.from_quat(rG_left) * r_left.inv()).magnitude() ** 2
        )
        lbg.append(0.0)
        ubg.append(INFTY)

        tf_right = self.right_serial_chain.forward_kinematics(q_right)

        p_right = tf_right.translation().as_vector()
        g.append(p_err_max - cs.sumsqr(p_right - pG_right))
        lbg.append(0.0)
        ubg.append(INFTY)

        r_right = tf_right.rotation()
        g.append(
            r_err_max
            - (sc.Rotation.from_quat(rG_right) * r_right.inv()).magnitude() ** 2
        )
        lbg.append(0.0)
        ubg.append(INFTY)

        # Joint limits
        joint_limits = {}
        left_arm_limits = self.left_serial_chain.get_joint_limits()
        right_arm_limits = self.right_serial_chain.get_joint_limits()
        for j, n in enumerate(self.left_serial_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = left_arm_limits[j]
        for j, n in enumerate(self.right_serial_chain.get_joint_parameter_names()):
            if n not in joint_limits:
                joint_limits[n] = right_arm_limits[j]

        for j, n in enumerate(self.joint_names):
            g.append(q[j])
            lbg.append(joint_limit_safety * joint_limits[n].lower)
            ubg.append(joint_limit_safety * joint_limits[n].upper)

        # Collision avoidance with collision spheres (assumes start/end are collision-free)

        q_dict = {n: q[j] for j, n in enumerate(self.joint_names)}

        Tf = self.chain.forward_kinematics(q_dict)  # transforms for all links in robot

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
        x = q
        p = cs.vertcat(
            pG_left,
            rG_left,
            pG_right,
            rG_right,
            qn,
            p_err_max,
            r_err_max,
            maintain_gaze,
        )
        for colsphere in collision_spheres:
            p = cs.vertcat(p, colsphere.radius, colsphere.position)

        g = vectorize(g)

        # Setup NLP
        prob = {"x": x, "p": p, "f": cost, "g": g}
        self.solver = cs.nlpsol("solver", "ipopt", prob)
        self.lbg = vectorize(lbg)
        self.ubg = vectorize(ubg)

    def reset(self, config):
        self.config = config
        x0 = [config["q0"]]
        p = [
            config["pG_left"],
            config["rG_left"],
            config["pG_right"],
            config["rG_right"],
            config["q0"],  # use qn = q0
            config["p_err_max"],
            config["r_err_max"],
            config["maintain_gaze"],
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

    def solve(self):
        self.solution = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        return stats["success"]

    def get_solution(self):
        qsol = self.solution["x"].toarray().flatten().tolist()
        return {n: q for n, q in zip(self.joint_names, qsol)}


class DualArmIKController(Program):

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
        pG_left = cs.SX.sym("pG_left", 3)  # position goal for left hand
        pG_right = cs.SX.sym("pG_right", 3)  # position goal for right hand
        rG_left = cs.SX.sym("rG_left", 4)  # rotation goal for left hand
        rG_right = cs.SX.sym("rG_right", 4)  # rotation goal for right hand
        q = cs.SX.sym("q", self.dof)  # current joint positions
        qn = cs.SX.sym("qn", self.dof)  # current joint positions

        w_dq = cs.SX.sym("w_dq")  # weight on the joint velocities
        w_p = cs.SX.sym("w_p")  # weight on eff position
        w_r = cs.SX.sym("w_r")  # weight on eff rotation
        w_gaze = cs.SX.sym("w_gaze")  # weight on gaze constraint
        w_qn = cs.SX.sym("w_qn")  # weight nominal configuration

        # Cost: joint velocity
        cost += w_dq * cs.sumsqr(dq)

        # Other vars
        qnext = q + dt * dq

        # Extract left/right
        q_from_name = lambda n: q[self.joint_names.index(n)]
        q_left = cs.vertcat(
            *[q_from_name(n) for n in self.left_arm_chain.get_joint_parameter_names()]
        )
        q_right = cs.vertcat(
            *[q_from_name(n) for n in self.right_arm_chain.get_joint_parameter_names()]
        )

        qnext_from_name = lambda n: qnext[self.joint_names.index(n)]
        qnext_left = cs.vertcat(
            *[
                qnext_from_name(n)
                for n in self.left_arm_chain.get_joint_parameter_names()
            ]
        )
        qnext_right = cs.vertcat(
            *[
                qnext_from_name(n)
                for n in self.right_arm_chain.get_joint_parameter_names()
            ]
        )

        dq_from_name = lambda n: dq[self.joint_names.index(n)]
        dq_left = cs.vertcat(
            *[dq_from_name(n) for n in self.left_arm_chain.get_joint_parameter_names()]
        )
        dq_right = cs.vertcat(
            *[dq_from_name(n) for n in self.right_arm_chain.get_joint_parameter_names()]
        )

        # Current FK/jac for each arm
        fk_left = self.left_arm_chain.forward_kinematics(q_left)
        fk_right = self.right_arm_chain.forward_kinematics(q_right)

        J_left = self.left_arm_chain.jacobian(q_left)
        J_right = self.right_arm_chain.jacobian(q_right)

        # Cost: eff position
        p_left = fk_left.translation().as_vector()
        dp_left = J_left[:3, :] @ dq_left
        pnext_left = p_left + dt * dp_left
        cost += w_p * cs.sumsqr(pG_left - pnext_left)

        p_right = fk_right.translation().as_vector()
        dp_right = J_right[:3, :] @ dq_right
        pnext_right = p_right + dt * dp_right
        cost += w_p * cs.sumsqr(pG_right - pnext_right)

        # Cost: rotation
        RG_left = sc.Rotation.from_quat(rG_left)
        R_left = fk_left.rotation()
        R_error_current_left = RG_left.as_matrix() @ R_left.as_matrix().T
        theta_error_current_left = (
            cs.vertcat(
                R_error_current_left[2, 1] - R_error_current_left[1, 2],
                R_error_current_left[0, 2] - R_error_current_left[2, 0],
                R_error_current_left[1, 0] - R_error_current_left[0, 1],
            )
            / 2.0
        )
        omega_left = J_left[3:, :] @ dq_left
        theta_error_next_left = theta_error_current_left - dt * omega_left
        cost += w_r * cs.sumsqr(theta_error_next_left)

        RG_right = sc.Rotation.from_quat(rG_right)
        R_right = fk_right.rotation()
        R_error_current_right = RG_right.as_matrix() @ R_right.as_matrix().T
        theta_error_current_right = (
            cs.vertcat(
                R_error_current_right[2, 1] - R_error_current_right[1, 2],
                R_error_current_right[0, 2] - R_error_current_right[2, 0],
                R_error_current_right[1, 0] - R_error_current_right[0, 1],
            )
            / 2.0
        )
        omega_right = J_right[3:, :] @ dq_right
        theta_error_next_right = theta_error_current_right - dt * omega_right
        cost += w_r * cs.sumsqr(theta_error_next_right)

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

        # Gaze
        zed_chain = self.chain.get_serial("zed_link")

        q_from_name = lambda n: q[self.joint_names.index(n)]
        q_zed = cs.vertcat(
            *[q_from_name(n) for n in zed_chain.get_joint_parameter_names()]
        )

        dq_from_name = lambda n: dq[self.joint_names.index(n)]
        dq_zed = cs.vertcat(
            *[dq_from_name(n) for n in zed_chain.get_joint_parameter_names()]
        )

        gaze_goal = 0.5 * (p_left + p_right)

        def gaze_funcs():
            q_sym = cs.SX.sym("q", zed_chain.dof)
            g = cs.SX.sym("g", 3)
            T = zed_chain.forward_kinematics(q_sym)
            e = T.translation().as_vector()
            z = T.rotation().as_matrix()[:, 2]
            alpha = cs.dot(g - e, z)
            a = e + alpha * z
            d2 = cs.sumsqr(a - g)
            J = cs.jacobian(d2, q_sym)
            f_fun = cs.Function("f", [q_sym, g], [d2])
            J_fun = cs.Function("J", [q_sym, g], [J])
            return f_fun, J_fun

        f_fun, J_fun = gaze_funcs()
        cost += w_gaze * cs.sumsqr(
            f_fun(q_zed, gaze_goal) + dt * J_fun(q_zed, gaze_goal) @ dq_zed
        )

        # Cost: Nominal configuration
        cost += w_qn * cs.sumsqr(qnext - qn)

        # Stable base constraint
        q_dict = {n: qnext[j] for j, n in enumerate(self.joint_names)}
        dq_dict = {n: dq[j] for j, n in enumerate(self.joint_names)}
        lbg_, g_, ubg_ = self.stable_robot_constraint_linearized(q_dict, dq_dict, dt)

        lbg.append(lbg_)
        g.append(g_)
        ubg.append(ubg_)

        # Vectorize decision variables and parameters and constraints
        x = dq
        p = cs.vertcat(
            *[
                pG_left,
                pG_right,
                rG_left,
                rG_right,
                q,
                w_dq,
                w_p,
                w_r,
                w_gaze,
                qn,
                w_qn,
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
            config["pG_left"],
            config["pG_right"],
            config["rG_left"],
            config["rG_right"],
            config["q"],
            config["w_dq"],
            config["w_p"],
            config["w_r"],
            config["w_gaze"],
            config["qn"],
            config["w_qn"],
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
