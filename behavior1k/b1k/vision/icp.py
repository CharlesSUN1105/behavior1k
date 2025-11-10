"""
ICP (Iterative Closest Point) matching module.

Provides functionality for:
- Pose estimation using ICP
- Coarse-to-fine alignment
- Fitness evaluation
"""

from typing import Tuple, Optional
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class ICPMatcher:
    """
    ICP-based pose estimator.
    
    Uses iterative closest point algorithm to estimate object pose
    by aligning observed point cloud with model point cloud.
    """
    
    def __init__(
        self,
        max_correspondence_distance: float = 0.05,
        max_iterations: int = 50,
        use_point_to_plane: bool = False,
    ):
        """
        Initialize ICP matcher.
        
        Args:
            max_correspondence_distance: Maximum distance for correspondences (meters)
            max_iterations: Maximum ICP iterations
            use_point_to_plane: Use point-to-plane ICP (requires normals)
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for ICP")
        
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations
        self.use_point_to_plane = use_point_to_plane
    
    def estimate_pose(
        self,
        observed_pcd: "o3d.geometry.PointCloud",
        model_pcd: "o3d.geometry.PointCloud",
        initial_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate object pose using ICP.
        
        Args:
            observed_pcd: Observed point cloud from sensor
            model_pcd: Model point cloud from mesh
            initial_transform: Initial 4x4 transformation guess
        
        Returns:
            Tuple of (transformation_matrix, fitness_score)
                - transformation_matrix: 4x4 pose matrix
                - fitness_score: 0-1 quality metric (higher is better)
        """
        return estimate_pose_icp(
            observed_pcd,
            model_pcd,
            initial_transform,
            self.max_correspondence_distance,
            self.max_iterations,
            self.use_point_to_plane,
        )
    
    def estimate_pose_with_coarse_alignment(
        self,
        observed_pcd: "o3d.geometry.PointCloud",
        model_pcd: "o3d.geometry.PointCloud",
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate pose with automatic coarse alignment.
        
        Uses RANSAC for coarse alignment before ICP refinement.
        
        Args:
            observed_pcd: Observed point cloud
            model_pcd: Model point cloud
        
        Returns:
            Tuple of (transformation_matrix, fitness_score)
        """
        # Perform coarse alignment using RANSAC
        initial_transform = coarse_alignment_ransac(observed_pcd, model_pcd)
        
        # Refine with ICP
        return self.estimate_pose(observed_pcd, model_pcd, initial_transform)


def estimate_pose_icp(
    observed_pcd: "o3d.geometry.PointCloud",
    model_pcd: "o3d.geometry.PointCloud",
    initial_transform: Optional[np.ndarray] = None,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 50,
    use_point_to_plane: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Estimate object pose using ICP algorithm.
    
    Args:
        observed_pcd: Observed point cloud from camera
        model_pcd: Model point cloud from mesh
        initial_transform: Initial 4x4 transformation (identity if None)
        max_correspondence_distance: Max distance for point matching (meters)
        max_iterations: Maximum ICP iterations
        use_point_to_plane: Use point-to-plane metric (requires normals)
    
    Returns:
        Tuple of:
            - transformation: 4x4 transformation matrix
            - fitness: Matching quality score (0-1)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required")
    
    # Use identity transform if no initial guess provided
    if initial_transform is None:
        initial_transform = np.eye(4)
    
    # Ensure normals are computed if using point-to-plane
    if use_point_to_plane:
        if not observed_pcd.has_normals():
            observed_pcd.estimate_normals()
        if not model_pcd.has_normals():
            model_pcd.estimate_normals()
        
        estimation_method = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
    else:
        estimation_method = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
    
    # Run ICP
    result = o3d.pipelines.registration.registration_icp(
        source=observed_pcd,
        target=model_pcd,
        max_correspondence_distance=max_correspondence_distance,
        init=initial_transform,
        estimation_method=estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )
    
    return result.transformation, result.fitness


def coarse_alignment_ransac(
    source_pcd: "o3d.geometry.PointCloud",
    target_pcd: "o3d.geometry.PointCloud",
    voxel_size: float = 0.05,
) -> np.ndarray:
    """
    Perform coarse alignment using FPFH features and RANSAC.
    
    Args:
        source_pcd: Source point cloud
        target_pcd: Target point cloud
        voxel_size: Voxel size for feature computation
    
    Returns:
        4x4 initial transformation matrix
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required")
    
    # Downsample for feature extraction
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    
    # RANSAC registration
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=(
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100000, confidence=0.999
        ),
    )
    
    return result.transformation


def evaluate_registration(
    source_pcd: "o3d.geometry.PointCloud",
    target_pcd: "o3d.geometry.PointCloud",
    transformation: np.ndarray,
    max_correspondence_distance: float = 0.05,
) -> Tuple[float, float]:
    """
    Evaluate registration quality.
    
    Args:
        source_pcd: Source point cloud
        target_pcd: Target point cloud
        transformation: Transformation to evaluate
        max_correspondence_distance: Distance threshold for correspondences
    
    Returns:
        Tuple of (fitness, rmse)
            - fitness: Ratio of inlier correspondences (0-1)
            - rmse: Root mean square error of inliers
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required")
    
    result = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, max_correspondence_distance, transformation
    )
    
    return result.fitness, result.inlier_rmse

