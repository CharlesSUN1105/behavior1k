"""
Camera module for RGBD data capture from PyBullet simulator.

This module provides functionality to:
- Capture RGB and depth images
- Extract camera parameters (intrinsics, extrinsics)
- Handle camera coordinate transformations
"""

from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pybullet_utils.bullet_client import BulletClient


class RGBDCamera:
    """
    RGBD camera for capturing images from PyBullet simulator.
    
    This class manages camera parameters and provides methods to capture
    RGBD data with proper calibration information.
    """
    
    def __init__(
        self,
        client: "BulletClient",
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near: float = 0.01,
        far: float = 10.0,
    ):
        """
        Initialize RGBD camera.
        
        Args:
            client: PyBullet client instance
            width: Image width in pixels
            height: Image height in pixels
            fov: Field of view in degrees
            near: Near clipping plane distance (meters)
            far: Far clipping plane distance (meters)
        """
        self.client = client
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        
        # Compute camera intrinsic matrix
        self.intrinsic_matrix = self._compute_intrinsic_matrix()
    
    def _compute_intrinsic_matrix(self) -> np.ndarray:
        """
        Compute camera intrinsic matrix from FOV and image dimensions.
        
        Returns:
            3x3 intrinsic matrix
        """
        fov_rad = np.radians(self.fov)
        fx = self.width / (2.0 * np.tan(fov_rad / 2.0))
        fy = self.height / (2.0 * np.tan(fov_rad / 2.0))
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return intrinsic
    
    def capture(
        self,
        camera_pos: Tuple[float, float, float],
        camera_target: Tuple[float, float, float],
        camera_up: Tuple[float, float, float] = (0, 0, 1),
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """
        Capture RGBD image from current viewpoint.
        
        Args:
            camera_pos: Camera position [x, y, z] in world frame
            camera_target: Camera target point [x, y, z] in world frame
            camera_up: Camera up vector [x, y, z]
        
        Returns:
            Tuple of:
                - rgb_image: RGB image array (H, W, 3), uint8
                - depth_image: Depth image array (H, W), float32 in meters
                - view_matrix: 4x4 view matrix
                - projection_matrix: 4x4 projection matrix
        """
        # Compute view matrix
        view_matrix = self.client.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=camera_up
        )
        
        # Compute projection matrix
        projection_matrix = self.client.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far
        )
        
        # Capture camera image
        width, height, rgb_img, depth_img, seg_img = self.client.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=self.client.ER_BULLET_HARDWARE_OPENGL  # Use hardware rendering
        )
        
        # Process RGB image
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))[:, :, :3]  # Remove alpha
        
        # Process depth image (convert from normalized to meters)
        depth_buffer = np.array(depth_img, dtype=np.float32).reshape((height, width))
        depth_array = self.far * self.near / (
            self.far - (self.far - self.near) * depth_buffer
        )
        
        # Convert matrices to numpy arrays
        view_matrix_np = np.array(view_matrix).reshape((4, 4), order='F')
        projection_matrix_np = np.array(projection_matrix).reshape((4, 4), order='F')
        
        return rgb_array, depth_array, view_matrix_np, projection_matrix_np
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Get camera intrinsic matrix.
        
        Returns:
            3x3 intrinsic matrix
        """
        return self.intrinsic_matrix.copy()


def capture_rgbd(
    client: "BulletClient",
    camera_pos: Tuple[float, float, float],
    camera_target: Tuple[float, float, float],
    camera_up: Tuple[float, float, float] = (0, 0, 1),
    width: int = 640,
    height: int = 480,
    fov: float = 60.0,
    near: float = 0.01,
    far: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to capture RGBD data.
    
    Args:
        client: PyBullet client instance
        camera_pos: Camera position [x, y, z]
        camera_target: Camera target point [x, y, z]
        camera_up: Camera up vector [x, y, z]
        width: Image width in pixels
        height: Image height in pixels
        fov: Field of view in degrees
        near: Near clipping plane
        far: Far clipping plane
    
    Returns:
        Tuple of (rgb_image, depth_image, intrinsic_matrix)
    """
    camera = RGBDCamera(client, width, height, fov, near, far)
    rgb, depth, _, _ = camera.capture(camera_pos, camera_target, camera_up)
    intrinsic = camera.get_intrinsic_matrix()
    
    return rgb, depth, intrinsic

