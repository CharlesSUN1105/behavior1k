"""
Grasp pose adjustment module.

Adjusts pre-defined grasp poses based on detected object poses.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from pybullet_utils.bullet_client import BulletClient

from .camera import RGBDCamera
from .segmentation import ObjectSegmentor, capture_segmentation_image
from .pointcloud import PointCloudProcessor, load_mesh_as_pointcloud
from .icp import ICPMatcher
from .transforms import CoordinateTransformer


class GraspPoseAdjuster:
    """
    Automatic grasp pose adjuster using vision.
    
    Detects object pose using RGBD+ICP and adjusts pre-defined grasp poses
    to match the detected object position and orientation.
    """
    
    def __init__(
        self,
        client: "BulletClient",
        robot_id: int,
        object_mesh_path: str,
        camera_link_name: str = "zed_link",
        use_sam: bool = False,
        sam_checkpoint: Optional[str] = None,
    ):
        """
        Initialize pose adjuster.
        
        Args:
            client: PyBullet client
            robot_id: Robot body ID
            object_mesh_path: Path to object mesh file
            camera_link_name: Name of camera link on robot (default "zed_link")
            use_sam: Use SAM for segmentation
            sam_checkpoint: Path to SAM checkpoint (if use_sam=True)
        """
        self.client = client
        self.robot_id = robot_id
        self.camera_link_name = camera_link_name
        
        # Initialize components
        self.camera = RGBDCamera(client)
        self.segmentor = ObjectSegmentor(use_sam, sam_checkpoint)
        self.pointcloud_processor = PointCloudProcessor()
        self.icp_matcher = ICPMatcher()
        self.transformer = CoordinateTransformer(client, robot_id)
        
        # Load object model
        self.object_mesh_pcd = load_mesh_as_pointcloud(object_mesh_path)
        self.object_mesh_pcd = self.pointcloud_processor.preprocess(
            self.object_mesh_pcd
        )
    
    def detect_object_pose(
        self,
        camera_pos: tuple,
        camera_target: tuple,
        object_id: Optional[int] = None,
        prompt_point: Optional[tuple] = None,
        camera_up: tuple = (0.0, 0.0, 1.0),
    ) -> tuple[np.ndarray, float]:
        """
        Detect object pose in robot base frame.
        
        Args:
            camera_pos: Camera position in world frame
            camera_target: Camera target in world frame
            object_id: Object body ID (for PyBullet segmentation)
            prompt_point: SAM prompt point (x, y) if using SAM
        
        Returns:
            Tuple of (pose_matrix, fitness_score)
                - pose_matrix: 4x4 object pose in base frame
                - fitness_score: ICP matching quality (0-1)
        """
        # 1. Capture RGBD
        rgb, depth, view_matrix, _ = self.camera.capture(
            camera_pos, camera_target, camera_up
        )
        intrinsic = self.camera.get_intrinsic_matrix()
        camera_pose_world = np.linalg.inv(view_matrix)
        
        # 2. Segment object
        if object_id is not None:
            # Use PyBullet segmentation
            seg_image = capture_segmentation_image(
                self.client,
                camera_pos,
                camera_target,
                camera_up=camera_up,
                width=self.camera.width,
                height=self.camera.height,
            )
            mask = self.segmentor.segment(
                rgb, method="pybullet", object_id=object_id, seg_image=seg_image
            )
        else:
            # Use SAM
            mask = self.segmentor.segment(
                rgb, method="sam", prompt_point=prompt_point
            )
        
        # 3. Extract point cloud
        observed_pcd = self.pointcloud_processor.extract_from_rgbd(
            rgb, depth, mask, intrinsic
        )
        observed_pcd = self.pointcloud_processor.preprocess(observed_pcd)

        if observed_pcd.is_empty():
            raise ValueError(
                "No points extracted for ICP – check segmentation mask and depth data"
            )
        
        # 4. ICP matching
        initial_transform = self._compute_initial_transform(observed_pcd)
        pose_camera, fitness = self.icp_matcher.estimate_pose(
            observed_pcd,
            self.object_mesh_pcd,
            initial_transform=initial_transform,
        )
        
        # 5. Transform to base frame
        pose_base = self.transformer.transform_pose_to_base(
            pose_camera,
            self.camera_link_name,
            camera_pose_world,
        )
        
        return pose_base, fitness

    def _compute_initial_transform(
        self, observed_pcd: "o3d.geometry.PointCloud"
    ) -> np.ndarray:
        """Build a coarse initial transform by aligning centroids."""
        obs_center = np.asarray(observed_pcd.get_center())
        model_center = np.asarray(self.object_mesh_pcd.get_center())

        transform = np.eye(4)
        transform[:3, 3] = obs_center - model_center
        return transform
    
    def adjust_grasp_pose(
        self,
        base_grasp_pose: Dict[str, Any],
        detected_object_pose: np.ndarray,
        base_object_position: tuple = (0.0, 0.0, 0.0),
    ) -> Dict[str, Any]:
        """
        Adjust grasp pose based on detected object pose.
        
        Args:
            base_grasp_pose: Base grasp pose dict with "base_position" and "joints"
            detected_object_pose: 4x4 detected object pose in base frame
            base_object_position: Assumed object position in base grasp pose
        
        Returns:
            Adjusted grasp pose dict
        """
        return adjust_grasp_pose(
            base_grasp_pose, detected_object_pose, base_object_position
        )


def adjust_grasp_pose(
    base_grasp_pose: Dict[str, Any],
    detected_object_pose: np.ndarray,
    base_object_position: tuple = (0.0, 0.0, 0.0),
) -> Dict[str, Any]:
    """
    Adjust grasp pose based on detected object position.
    
    The base grasp pose assumes the object is at base_object_position.
    This function computes the offset and adjusts the robot base position.
    
    Args:
        base_grasp_pose: Dict with "base_position" and "joints"
        detected_object_pose: 4x4 transformation of detected object
        base_object_position: Object position assumed in base_grasp_pose
    
    Returns:
        Adjusted grasp pose dict
    """
    # Extract detected object position
    detected_position = detected_object_pose[:3, 3]
    
    # Compute position offset
    base_obj_pos = np.array(base_object_position)
    position_offset = detected_position - base_obj_pos
    
    # Adjust robot base position
    base_robot_pos = np.array(base_grasp_pose["base_position"])
    adjusted_robot_pos = base_robot_pos + position_offset
    
    # Create adjusted pose
    adjusted_pose = {
        "base_position": adjusted_robot_pos.tolist(),
        "joints": base_grasp_pose["joints"].copy(),
    }
    
    return adjusted_pose


def adjust_grasp_pose_with_rotation(
    base_grasp_pose: Dict[str, Any],
    detected_object_pose: np.ndarray,
    base_object_pose: np.ndarray = np.eye(4),
) -> Dict[str, Any]:
    """
    Adjust grasp pose considering both position and rotation.
    
    This is more complex and may require IK to recompute joint angles.
    
    Args:
        base_grasp_pose: Base grasp pose
        detected_object_pose: Detected object 4x4 pose
        base_object_pose: Object pose assumed in base grasp pose
    
    Returns:
        Adjusted grasp pose
    """
    # Compute relative transform
    # T_offset = T_detected @ T_base^-1
    base_obj_inv = np.linalg.inv(base_object_pose)
    transform_offset = detected_object_pose @ base_obj_inv
    
    # Apply offset to robot base
    base_robot_pos = np.array(base_grasp_pose["base_position"])
    base_robot_pose = np.eye(4)
    base_robot_pose[:3, 3] = base_robot_pos
    
    adjusted_robot_pose = transform_offset @ base_robot_pose
    adjusted_robot_pos = adjusted_robot_pose[:3, 3]
    
    # TODO: If rotation is significant, need to:
    # 1. Adjust base orientation or
    # 2. Use IK to recompute joint angles
    
    adjusted_pose = {
        "base_position": adjusted_robot_pos.tolist(),
        "joints": base_grasp_pose["joints"].copy(),
    }
    
    return adjusted_pose
