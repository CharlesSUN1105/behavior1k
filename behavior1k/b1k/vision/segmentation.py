"""
Object segmentation module.

Provides methods for segmenting objects from RGB images using:
- SAM (Segment Anything Model)
- PyBullet's built-in segmentation
"""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pybullet_utils.bullet_client import BulletClient


class ObjectSegmentor:
    """
    Object segmentation using multiple backends.
    
    Supports both SAM and PyBullet segmentation methods.
    """
    
    def __init__(self, use_sam: bool = False, sam_checkpoint: Optional[str] = None):
        """
        Initialize segmentor.
        
        Args:
            use_sam: Whether to use SAM for segmentation
            sam_checkpoint: Path to SAM checkpoint file (if use_sam=True)
        """
        self.use_sam = use_sam
        self.sam_model = None
        
        if use_sam:
            self._load_sam_model(sam_checkpoint)
    
    def _load_sam_model(self, checkpoint_path: Optional[str]):
        """
        Load SAM model.
        
        Args:
            checkpoint_path: Path to SAM checkpoint
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if checkpoint_path is None:
                raise ValueError("SAM checkpoint path is required when use_sam=True")
            
            # Load SAM model (default to vit_h variant)
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            self.sam_model = SamPredictor(sam)
            print(f"✓ SAM model loaded from {checkpoint_path}")
            
        except ImportError:
            raise ImportError(
                "segment_anything package not found. "
                "Install with: pip install segment-anything"
            )
    
    def segment(
        self,
        rgb_image: np.ndarray,
        method: str = "auto",
        prompt_point: Optional[Tuple[int, int]] = None,
        object_id: Optional[int] = None,
        seg_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Segment object from RGB image.
        
        Args:
            rgb_image: RGB image array (H, W, 3)
            method: Segmentation method ("sam", "pybullet", or "auto")
            prompt_point: Point prompt for SAM (x, y pixel coordinates)
            object_id: Object ID for PyBullet segmentation
            seg_image: PyBullet segmentation image
        
        Returns:
            Binary mask array (H, W) with True for object pixels
        """
        if method == "auto":
            method = "sam" if self.use_sam else "pybullet"
        
        if method == "sam":
            return segment_object_sam(rgb_image, self.sam_model, prompt_point)
        elif method == "pybullet":
            if seg_image is None or object_id is None:
                raise ValueError("seg_image and object_id required for pybullet method")
            return segment_object_pybullet(seg_image, object_id)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")


def segment_object_sam(
    rgb_image: np.ndarray,
    sam_model,
    prompt_point: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Segment object using SAM.
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        sam_model: SAM predictor model
        prompt_point: Optional point prompt (x, y)
    
    Returns:
        Binary mask (H, W)
    """
    if sam_model is None:
        raise ValueError("SAM model not loaded")
    
    sam_model.set_image(rgb_image)
    
    if prompt_point is not None:
        # Use point prompt
        masks, scores, _ = sam_model.predict(
            point_coords=np.array([prompt_point]),
            point_labels=np.array([1]),  # 1 = foreground
            multimask_output=False
        )
        return masks[0]
    else:
        # Automatic segmentation - return largest mask
        masks = sam_model.generate(rgb_image)
        if len(masks) == 0:
            return np.zeros(rgb_image.shape[:2], dtype=bool)
        
        # Select largest mask
        largest_mask = max(masks, key=lambda m: m["segmentation"].sum())
        return largest_mask["segmentation"]


def segment_object_pybullet(seg_image: np.ndarray, object_id: int) -> np.ndarray:
    """
    Segment object using PyBullet's segmentation image.
    
    Args:
        seg_image: Segmentation image from PyBullet (H, W)
        object_id: Target object's body ID
    
    Returns:
        Binary mask (H, W)
    """
    seg_array = np.asarray(seg_image, dtype=np.int32)
    object_ids = decode_pybullet_object_ids(seg_array)
    mask = object_ids == int(object_id)

    if not np.any(mask):
        raise ValueError(
            f"PyBullet segmentation mask is empty for object_id={object_id}. "
            "Check camera pose or ensure the object is visible."
        )

    return mask


def capture_segmentation_image(
    client: "BulletClient",
    camera_pos: Tuple[float, float, float],
    camera_target: Tuple[float, float, float],
    camera_up: Tuple[float, float, float] = (0, 0, 1),
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """
    Capture segmentation image from PyBullet.
    
    Args:
        client: PyBullet client
        camera_pos: Camera position
        camera_target: Camera target
        camera_up: Camera up vector
        width: Image width
        height: Image height
    
    Returns:
        Segmentation image where each pixel value is the object ID
    """
    view_matrix = client.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=camera_target,
        cameraUpVector=camera_up
    )
    
    projection_matrix = client.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.01,
        farVal=10.0
    )
    
    _, _, _, _, seg_img = client.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        flags=client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    
    seg_array = np.array(seg_img, dtype=np.int32).reshape((height, width))
    return seg_array


def decode_pybullet_object_ids(seg_image: np.ndarray) -> np.ndarray:
    """Decode objectUniqueId from PyBullet segmentation image."""
    seg_uint = np.asarray(seg_image, dtype=np.uint32)
    obj_mask = np.uint32((1 << 24) - 1)
    obj_ids = seg_uint & obj_mask

    # Background pixels are encoded as -1 -> 0xFFFFFF, restore to -1 for readability
    background_value = obj_mask
    obj_ids = obj_ids.astype(np.int32, copy=False)
    obj_ids[obj_ids == background_value] = -1
    return obj_ids
