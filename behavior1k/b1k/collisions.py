from dataclasses import dataclass
from typing import Optional
import casadi as cs


@dataclass
class CollisionSphere:
    name: str
    radius: cs.SX | float
    position: cs.SX | list[float]


@dataclass
class CollisionLink:
    base_link_name: str
    radius: float
    color: Optional[list[float]] = None
    offset: Optional[list[float]] = None


def collision_links_with_base_link(link_name: str, ls: list[CollisionLink]):
    return [cl for cl in ls if cl.base_link_name == link_name]


# Define collision link groups for r1pro

red = (1.0, 0.0, 0.0, 0.4)
green = (0.0, 1.0, 0.0, 0.4)
blue = (0.0, 0.0, 1.0, 0.4)
orange = (1.0, 165.0 / 255.0, 0.0, 0.4)
yellow = (1.0, 1.0, 0.0, 0.4)
cyan = (0.0, 1.0, 1.0, 0.4)
magenta = (1.0, 0.0, 1.0, 0.4)

collision_links_base = [
    CollisionLink("base_link", 0.4, red, [0.0, 0.0, 0.1]),
    CollisionLink("steer_motor_link1", 0.1, red, [0.0, 0.0, 0.1]),
    CollisionLink("steer_motor_link2", 0.1, red, [0.0, 0.0, 0.1]),
    CollisionLink("steer_motor_link3", 0.1, red, [0.0, 0.0, 0.11]),
    CollisionLink("steer_motor_link3", 0.1, red, [0.0, 0.1, 0.11]),
    CollisionLink("steer_motor_link3", 0.1, red, [0.0, -0.1, 0.11]),
]

collision_links_torso = [
    CollisionLink("torso_link2", 0.125, green),
    CollisionLink("torso_link2", 0.125, green, [0, 0, 0.125]),
    CollisionLink("torso_link3", 0.125, green),
    CollisionLink("torso_link4", 0.15, green, [0.0, 0.0, 0.125]),
]

collision_links_head = [
    CollisionLink("zed_link", 0.135, orange, [0.06, 0.0, -0.075]),
]

collision_links_left_arm = [
    CollisionLink("left_arm_link1", 0.08, blue),
    CollisionLink("left_arm_link2", 0.075, blue, [-0.025, 0.0, 0.0]),
    CollisionLink("left_arm_link3", 0.06, blue),
    CollisionLink("left_arm_link3", 0.06, blue, [0.0, 0.0, -0.075]),
    CollisionLink("left_arm_link4", 0.075, blue, [0.0, -0.03, 0.0]),
    CollisionLink("left_arm_link5", 0.075, blue, [0.0, 0.0, -0.1]),
    CollisionLink("left_arm_link5", 0.075, blue),
    CollisionLink("left_arm_link6", 0.075, blue, [0.0, 0.025, 0.0]),
]

collision_links_right_arm = [
    CollisionLink("right_arm_link1", 0.08, yellow),
    CollisionLink("right_arm_link2", 0.075, yellow, [-0.025, 0.0, 0.0]),
    CollisionLink("right_arm_link3", 0.06, yellow),
    CollisionLink("right_arm_link3", 0.06, yellow, [0.0, 0.0, -0.075]),
    CollisionLink("right_arm_link4", 0.075, yellow, [0.0, -0.03, 0.0]),
    CollisionLink("right_arm_link5", 0.075, yellow, [0.0, 0.0, -0.1]),
    CollisionLink("right_arm_link5", 0.075, yellow),
    CollisionLink("right_arm_link6", 0.075, yellow, [0.0, -0.025, 0.0]),
]

collision_links_left_hand = [
    CollisionLink("left_gripper_link", 0.08, cyan),
]

collision_links_right_hand = [
    CollisionLink("right_gripper_link", 0.08, magenta),
]

collision_links = []
collision_links += collision_links_base
collision_links += collision_links_torso
collision_links += collision_links_head
collision_links += collision_links_left_arm
collision_links += collision_links_right_arm
collision_links += collision_links_left_hand
collision_links += collision_links_right_hand
