"""
Coordinate transformation utilities.

Handles transformations between:
- Camera frame
- Robot base frame
- Object frames
"""

from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from pybullet_utils.bullet_client import BulletClient


class CoordinateTransformer:
    """
    Coordinate transformation manager.
    
    Handles conversions between different coordinate frames in the robot system.
    """
    
    def __init__(self, client: "BulletClient", robot_id: int):
        """
        Initialize transformer.
        
        Args:
            client: PyBullet client
            robot_id: Robot body ID
        """
        self.client = client
        self.robot_id = robot_id
    
    def get_camera_to_base_transform(
        self, camera_link_name: str
    ) -> np.ndarray:
        """
        Get transformation from camera frame to robot base frame.
        
        Args:
            camera_link_name: Name of the camera link
        
        Returns:
            4x4 transformation matrix
        """
        return get_camera_to_base_transform(
            self.client, self.robot_id, camera_link_name
        )
    
    def transform_pose_to_base(
        self,
        pose_camera: np.ndarray,
        camera_link_name: str,
        camera_pose_in_world: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Transform pose from camera frame to base frame.
        
        Args:
            pose_camera: 4x4 pose in camera frame
            camera_link_name: Camera link name
            camera_pose_in_world: Optional 4x4 transform of camera frame in world coords
        
        Returns:
            4x4 pose in base frame
        """
        if camera_pose_in_world is not None:
            base_position, base_orientation = self.client.getBasePositionAndOrientation(
                self.robot_id
            )
            base_in_world = pose_to_matrix(base_position, base_orientation)
            world_to_base = invert_transform(base_in_world)
            camera_to_base = world_to_base @ camera_pose_in_world
        else:
            camera_to_base = self.get_camera_to_base_transform(camera_link_name)
        return transform_to_robot_base(pose_camera, camera_to_base)


def get_camera_to_base_transform(
    client: "BulletClient", robot_id: int, camera_link_name: str
) -> np.ndarray:
    """
    Get transformation from camera to robot base frame.
    
    Args:
        client: PyBullet client
        robot_id: Robot body ID
        camera_link_name: Name of camera link
    
    Returns:
        4x4 transformation matrix from camera to base
    """
    # Get link index
    link_index = get_link_index_by_name(client, robot_id, camera_link_name)
    
    if link_index == -1:
        # Camera is on base link
        position, orientation = client.getBasePositionAndOrientation(robot_id)
    else:
        # Get link state
        link_state = client.getLinkState(
            robot_id, link_index, computeForwardKinematics=True
        )
        position = link_state[0]  # World position
        orientation = link_state[1]  # World orientation (quaternion)

    # Convert to transformation matrices
    camera_in_world = pose_to_matrix(position, orientation)
    base_position, base_orientation = client.getBasePositionAndOrientation(robot_id)
    base_in_world = pose_to_matrix(base_position, base_orientation)
    world_to_base = invert_transform(base_in_world)

    # T_base^camera = T_base^world @ T_world^camera
    transform = world_to_base @ camera_in_world
    
    return transform


def get_link_index_by_name(
    client: "BulletClient", robot_id: int, link_name: str
) -> int:
    """
    Get link index by name.
    
    Args:
        client: PyBullet client
        robot_id: Robot body ID
        link_name: Link name to search for
    
    Returns:
        Link index (-1 for base link, or >=0 for other links)
    """
    # Check base link
    base_name = client.getBodyInfo(robot_id)[0].decode('utf-8')
    if base_name == link_name:
        return -1
    
    # Check all joints/links
    num_joints = client.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = client.getJointInfo(robot_id, i)
        if joint_info[12].decode('utf-8') == link_name:  # Link name is at index 12
            return i
    
    raise ValueError(f"Link '{link_name}' not found in robot")


def transform_to_robot_base(
    pose_camera_frame: np.ndarray, camera_to_base_transform: np.ndarray
) -> np.ndarray:
    """
    Transform pose from camera frame to robot base frame.
    
    Args:
        pose_camera_frame: 4x4 pose matrix in camera frame
        camera_to_base_transform: 4x4 transformation from camera to base
    
    Returns:
        4x4 pose matrix in base frame
    """
    # T_base^object = T_base^camera @ T_camera^object
    pose_base_frame = camera_to_base_transform @ pose_camera_frame
    return pose_base_frame


def pose_to_matrix(
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Convert position and quaternion to 4x4 transformation matrix.
    
    Args:
        position: (x, y, z)
        orientation: Quaternion (x, y, z, w)
    
    Returns:
        4x4 transformation matrix
    """
    # Convert quaternion to rotation matrix
    rotation = Rotation.from_quat(orientation)
    rotation_matrix = rotation.as_matrix()
    
    # Build 4x4 transformation
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    
    return transform


def matrix_to_pose(
    matrix: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Convert 4x4 transformation matrix to position and quaternion.
    
    Args:
        matrix: 4x4 transformation matrix
    
    Returns:
        Tuple of (position, orientation)
            - position: (x, y, z)
            - orientation: Quaternion (x, y, z, w)
    """
    position = tuple(matrix[:3, 3])
    
    rotation = Rotation.from_matrix(matrix[:3, :3])
    orientation = tuple(rotation.as_quat())  # Returns (x, y, z, w)
    
    return position, orientation


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 transformation matrix.
    
    Args:
        transform: 4x4 transformation matrix
    
    Returns:
        Inverted 4x4 transformation matrix
    """
    # For homogeneous transformation:
    # T^-1 = [R^T  -R^T*t]
    #        [0      1    ]
    
    R = transform[:3, :3]
    t = transform[:3, 3]
    
    R_inv = R.T
    t_inv = -R_inv @ t
    
    transform_inv = np.eye(4)
    transform_inv[:3, :3] = R_inv
    transform_inv[:3, 3] = t_inv
    
    return transform_inv
