"""
Point cloud processing module.

Provides functionality for:
- Extracting point clouds from RGBD data
- Filtering and preprocessing point clouds
- Converting between different point cloud formats
"""

from typing import Optional
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3D not available. Install with: pip install open3d")


class PointCloudProcessor:
    """
    Point cloud processing utility class.
    
    Handles point cloud extraction, filtering, and preprocessing.
    """
    
    def __init__(self, voxel_size: float = 0.006, remove_outliers: bool = True):
        """
        Initialize processor.
        
        Args:
            voxel_size: Voxel size for downsampling (meters)
            remove_outliers: Whether to remove statistical outliers
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for point cloud processing")
        
        self.voxel_size = voxel_size
        self.remove_outliers = remove_outliers
    
    def extract_from_rgbd(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        mask: np.ndarray,
        intrinsic_matrix: np.ndarray,
    ) -> "o3d.geometry.PointCloud":
        """
        Extract point cloud from RGBD data.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W) in meters
            mask: Binary mask (H, W) for object region
            intrinsic_matrix: Camera intrinsic matrix (3, 3)
        
        Returns:
            Open3D point cloud object
        """
        return extract_object_pointcloud(
            rgb_image, depth_image, mask, intrinsic_matrix
        )
    
    def preprocess(
        self, pointcloud: "o3d.geometry.PointCloud"
    ) -> "o3d.geometry.PointCloud":
        """
        Preprocess point cloud (downsample, filter, compute normals).
        
        Args:
            pointcloud: Input point cloud
        
        Returns:
            Preprocessed point cloud
        """
        # Downsample
        pcd_down = pointcloud.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Remove outliers
        if self.remove_outliers:
            pcd_down, _ = pcd_down.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )
        
        return pcd_down


def extract_object_pointcloud(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsic_matrix: np.ndarray,
) -> "o3d.geometry.PointCloud":
    """
    Extract point cloud from RGBD data for masked region.
    
    Args:
        rgb_image: RGB image (H, W, 3), uint8
        depth_image: Depth image (H, W), float in meters
        mask: Binary mask (H, W), bool
        intrinsic_matrix: Camera intrinsic matrix (3, 3)
    
    Returns:
        Open3D point cloud
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    height, width = depth_image.shape
    
    # Extract camera parameters
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # Build point cloud
    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            if mask[v, u]:  # Only process masked pixels
                z = depth_image[v, u]
                if z > 0 and np.isfinite(z):  # Valid depth
                    # Backproject to 3D
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])
                    colors.append(rgb_image[v, u] / 255.0)
    
    if len(points) == 0:
        print("⚠️ No valid points extracted from mask region")
        return o3d.geometry.PointCloud()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return pcd


def load_mesh_as_pointcloud(
    mesh_path: str, num_points: int = 10000
) -> "o3d.geometry.PointCloud":
    """
    Load mesh file and convert to point cloud.
    
    Args:
        mesh_path: Path to mesh file (.obj, .ply, etc.)
        num_points: Number of points to sample
    
    Returns:
        Sampled point cloud
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required")
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    if not mesh.has_vertices():
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    
    # Sample points uniformly
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    return pcd

