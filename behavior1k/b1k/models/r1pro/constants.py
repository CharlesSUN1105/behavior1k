from math import radians

r1pro_init_joint_state_deg = {
    "torso_joint1": -30.0,
    "torso_joint2": 90.0,
    "torso_joint3": 60.0,
    "torso_joint4": 0.0,
    "left_arm_joint1": 0.0,
    "left_arm_joint2": 0.0,
    "left_arm_joint3": 0.0,
    "left_arm_joint4": -90.0,
    "left_arm_joint5": 0.0,
    "left_arm_joint6": 0.0,
    "left_arm_joint7": 0.0,
    "right_arm_joint1": 0.0,
    "right_arm_joint2": 0.0,
    "right_arm_joint3": 0.0,
    "right_arm_joint4": -90.0,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": 0.0,
    "right_arm_joint7": 0.0,
}

r1pro_init_joint_state = {}
for k, v in r1pro_init_joint_state_deg.items():
    r1pro_init_joint_state[k] = radians(v)

# Add gripper joints in open state (gripper joints are prismatic joints in meters, max value is 0.05)
r1pro_init_joint_state["left_gripper_finger_joint1"] = 0.05
r1pro_init_joint_state["left_gripper_finger_joint2"] = 0.05
r1pro_init_joint_state["right_gripper_finger_joint1"] = 0.05
r1pro_init_joint_state["right_gripper_finger_joint2"] = 0.05

r1pro_T_joint_state_deg = {
    "torso_joint1": 0.0,
    "torso_joint2": 0.0,
    "torso_joint3": 0.0,
    "torso_joint4": 0.0,
    "left_arm_joint1": 0.0,
    "left_arm_joint2": 90.0,
    "left_arm_joint3": 0.0,
    "left_arm_joint4": 0.0,
    "left_arm_joint5": 0.0,
    "left_arm_joint6": 0.0,
    "left_arm_joint7": 0.0,
    "right_arm_joint1": 0.0,
    "right_arm_joint2": -90.0,
    "right_arm_joint3": 0.0,
    "right_arm_joint4": 0.0,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": 0.0,
    "right_arm_joint7": 0.0,
}


r1pro_T_joint_state = {}
for k, v in r1pro_T_joint_state_deg.items():
    r1pro_T_joint_state[k] = radians(v)
