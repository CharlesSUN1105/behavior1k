from abc import ABC, abstractmethod
from b1k.casadi_utils import INFTY
from b1k.kinematics import (
    build_r1pro_chain,
    build_r1pro_casadi_pinocchio_model_and_data,
    cpin,
)

import casadi as cs


class Program:

    def __init__(self):
        self.chain = build_r1pro_chain()
        self.solution = None  # variable containing the solution
        self.config = None  # configuration for reset
        self.cmodel, self.cdata = build_r1pro_casadi_pinocchio_model_and_data()

    def stable_robot_constraint_linearized(
        self, qc: dict[str, cs.SX], dq: dict[str, cs.SX], dt: cs.SX
    ):
        """Linearized stability constraint in dq."""

        # Convert qc to vector - ensure we get the right size
        qc_vec = []
        dq_vec = []
        for n in self.cmodel.names[1:]:  # Skip 'universe'
            jid = self.cmodel.getJointId(n)
            jnt = self.cmodel.joints[jid]
            qc_vec.append(qc.get(n, cs.SX.zeros(jnt.nq)))
            dq_vec.append(dq.get(n, cs.SX.zeros(jnt.nq)))
        qc_sym = cs.vertcat(*qc_vec)
        dq_sym = cs.vertcat(*dq_vec)

        print(f"Debug: qc_sym size: {qc_sym.shape}, dq_sym size: {dq_sym.shape}")
        print(f"Debug: Model nv: {self.cmodel.nv}")

        # Current COM position and Jacobian
        com = cpin.centerOfMass(self.cmodel, self.cdata, qc_sym)
        J_com = cpin.jacobianCenterOfMass(self.cmodel, self.cdata, qc_sym)
        P_current = cs.vertcat(com[0], com[1])

        print(f"Debug: J_com size: {J_com.shape}")

        # Ensure J_com_xy has the right dimensions (2 x nv)
        J_com_xy = (
            J_com[:2, :] if J_com.shape[0] >= 2 else cs.SX.zeros(2, J_com.shape[1])
        )

        # Helper function to get frame position and Jacobian
        def frame_xy_and_jacobian(frame_name):
            fid = self.cmodel.getFrameId(frame_name)
            cpin.forwardKinematics(self.cmodel, self.cdata, qc_sym)
            cpin.updateFramePlacements(self.cmodel, self.cdata)

            p = self.cdata.oMf[fid].translation
            p_xy = cs.vertcat(p[0], p[1])

            # Compute frame Jacobian - this should return (6 x nv)
            J = cpin.computeFrameJacobian(self.cmodel, self.cdata, qc_sym, fid)
            print(f"Debug: {frame_name} Jacobian size: {J.shape}")

            # Take only XY components (2 x nv)
            J_xy = J[:2, :] if J.shape[0] >= 2 else cs.SX.zeros(2, J.shape[1])

            return p_xy, J_xy

        # Get current vertices and their Jacobians
        vertices = []
        vertex_jacobians = []
        for i in range(1, 4):
            v_xy, J_v = frame_xy_and_jacobian(f"steer_motor_joint{i}")
            vertices.append(v_xy)
            vertex_jacobians.append(J_v)

        # Cross product helper for vectors
        def cross(e, v):
            return e[0] * v[1] - e[1] * v[0]

        # Cross product helper for Jacobian rows
        def cross_jacobian_vector(J, v):
            """Compute cross product between each column of J and vector v"""
            # J should be (2 x nv), v should be (2 x 1)
            return J[0, :] * v[1] - J[1, :] * v[0]

        def cross_vector_jacobian(v, J):
            """Compute cross product between vector v and each column of J"""
            return v[0] * J[1, :] - v[1] * J[0, :]

        # Determine winding order
        area = 0
        n_vertices = len(vertices)
        for i in range(n_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n_vertices]
            area += v1[0] * v2[1] - v2[0] * v1[1]

        safety_margin = 0.01

        # Build linearized constraints
        constraints = []
        for i in range(n_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n_vertices]
            J_v1 = vertex_jacobians[i]
            J_v2 = vertex_jacobians[(i + 1) % n_vertices]

            edge = v2 - v1
            J_edge = J_v2 - J_v1

            to_point = P_current - v1
            J_to_point = J_com_xy - J_v1

            # Current cross product value
            cross_current = cross(edge, to_point)

            # Jacobian of cross product
            J_cross = cross_jacobian_vector(J_edge, to_point) + cross_vector_jacobian(
                edge, J_to_point
            )

            print(f"Debug: J_cross size: {J_cross.shape}")

            # Edge length for margin
            edge_length = cs.sqrt(edge[0] ** 2 + edge[1] ** 2)
            margin = safety_margin * edge_length

            # Ensure J_cross has the same number of columns as dq_sym has rows
            if J_cross.size2() != dq_sym.size1():
                print(
                    f"Warning: Dimension mismatch. J_cross: {J_cross.shape}, dq_sym: {dq_sym.shape}"
                )
                # Resize J_cross to match dq_sym
                if J_cross.size2() < dq_sym.size1():
                    # Pad with zeros
                    J_cross_padded = cs.SX.zeros(1, dq_sym.size1())
                    J_cross_padded[0, : J_cross.size2()] = J_cross
                    J_cross = J_cross_padded
                else:
                    # Truncate
                    J_cross = J_cross[0, : dq_sym.size1()]

            constraint = cs.mtimes(J_cross, dt * dq_sym) + cross_current - margin
            constraints.append(constraint)

        g = cs.vertcat(*constraints)

        # Set bounds
        if area > 0:  # CCW
            lbg = cs.DM.zeros(n_vertices)
            ubg = cs.DM([INFTY] * n_vertices)
        else:  # CW
            lbg = cs.DM([-INFTY] * n_vertices)
            ubg = cs.DM.zeros(n_vertices)

        return lbg, g, ubg

    def stable_robot_constraint(self, q: dict[str, cs.SX]):
        """Robust version with numerical safety margins and winding detection."""

        # Convert q to vector
        q_vec = []
        for n in self.cmodel.names[1:]:  # avoid 'universe' joint
            jid = self.cmodel.getJointId(n)
            jnt = self.cmodel.joints[jid]
            q_vec.append(q.get(n, cs.SX.zeros(jnt.nq)))
        q_sym = cs.vertcat(*q_vec)

        # Compute COM position
        com = cpin.centerOfMass(self.cmodel, self.cdata, q_sym)
        P = cs.vertcat(com[0], com[1])  # COM in XY plane

        # Helper function to get frame XY position
        def frame_xy(frame_name):
            fid = self.cmodel.getFrameId(frame_name)
            cpin.forwardKinematics(self.cmodel, self.cdata, q_sym)
            cpin.updateFramePlacements(self.cmodel, self.cdata)
            p = self.cdata.oMf[fid].translation
            return cs.vertcat(p[0], p[1])

        # Get support polygon vertices from steer motor joints
        vertices = [
            frame_xy("steer_motor_joint1"),
            frame_xy("steer_motor_joint2"),
            frame_xy("steer_motor_joint3"),
        ]

        # Cross product helper
        def cross(e, v):
            return e[0] * v[1] - e[1] * v[0]

        # Determine winding order by checking polygon area (shoelace formula)
        area = 0
        n_vertices = len(vertices)
        for i in range(n_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n_vertices]
            area += v1[0] * v2[1] - v2[0] * v1[1]

        # Add small safety margin to keep COM away from edges
        # This prevents numerical issues and provides stability margin
        safety_margin = 0.01  # 1cm safety margin

        # Create edge constraints with safety margin
        constraints = []
        for i in range(n_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n_vertices]
            edge = v2 - v1
            to_point = P - v1
            cross_val = cross(edge, to_point)

            # Apply safety margin based on edge length
            edge_length = cs.sqrt(edge[0] ** 2 + edge[1] ** 2)
            margin = safety_margin * edge_length

            constraints.append(cross_val - margin)

        g = cs.vertcat(*constraints)

        # Set bounds based on winding order with safety margin
        if area > 0:  # CCW
            lbg = cs.DM.zeros(n_vertices)
            ubg = cs.DM([INFTY] * n_vertices)
        else:  # CW
            lbg = cs.DM([-INFTY] * n_vertices)
            ubg = cs.DM.zeros(n_vertices)

        return lbg, g, ubg

    @abstractmethod
    def reset(self, config: dict):
        """Resets the solver."""

    @abstractmethod
    def solve(self) -> bool:
        """Solves the problem, returns success."""

    @abstractmethod
    def get_solution(self) -> dict:
        """Returns the solution for the last call to solve."""
