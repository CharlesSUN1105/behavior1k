"""
Vision module for object detection and pose estimation.

This module provides functionality for:
- RGBD data capture from PyBullet simulator
- Object segmentation using SAM or PyBullet segmentation
- Point cloud processing
- ICP-based pose estimation
- Dynamic grasp pose adjustment
"""

from .camera import RGBDCamera, capture_rgbd
from .segmentation import ObjectSegmentor, segment_object_sam, segment_object_pybullet
from .pointcloud import PointCloudProcessor, extract_object_pointcloud
from .icp import ICPMatcher, estimate_pose_icp
from .pose_adjuster import GraspPoseAdjuster, adjust_grasp_pose
from .transforms import CoordinateTransformer, transform_to_robot_base

__all__ = [
    # Camera
    "RGBDCamera",
    "capture_rgbd",
    # Segmentation
    "ObjectSegmentor",
    "segment_object_sam",
    "segment_object_pybullet",
    # Point Cloud
    "PointCloudProcessor",
    "extract_object_pointcloud",
    # ICP
    "ICPMatcher",
    "estimate_pose_icp",
    # Pose Adjustment
    "GraspPoseAdjuster",
    "adjust_grasp_pose",
    # Transforms
    "CoordinateTransformer",
    "transform_to_robot_base",
]

__version__ = "0.1.0"

