"""
Object configuration for vision-based manipulation.

Defines mesh paths and default parameters for each object.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

# Get objects directory
OBJECTS_DIR = Path(__file__).parent


@dataclass
class ObjectConfig:
    """Configuration for a manipulable object."""
    
    name: str
    mesh_path: Path
    default_position: Tuple[float, float, float]
    default_mass: float
    # Camera viewpoint for ICP
    camera_pos: Tuple[float, float, float]
    camera_target: Tuple[float, float, float]


# Object configurations
FRIDGE_CONFIG = ObjectConfig(
    name="fridge",
    mesh_path=OBJECTS_DIR / "fridge_dszchb.obj",
    default_position=(0.0, 0.0, 0.0),
    default_mass=0.0,  # Fixed
    camera_pos=(-1.5, 0.5, 1.0),
    camera_target=(0.0, 0.0, 0.8),
)

RADIO_CONFIG = ObjectConfig(
    name="radio",
    mesh_path=OBJECTS_DIR / "radio_wxnicr.obj",
    default_position=(0.0, 0.0, 0.82),  # On table
    default_mass=0.5,
    camera_pos=(0.5, 0.5, 1.0),
    camera_target=(0.0, 0.0, 0.82),
)

TRAY_CONFIG = ObjectConfig(
    name="tray",
    mesh_path=OBJECTS_DIR / "tray_gsxbym.obj",
    default_position=(0.0, 0.0, 0.891),
    default_mass=0.0,  # Fixed
    camera_pos=(0.5, 0.5, 1.2),
    camera_target=(0.0, 0.0, 0.891),
)

FRYING_PAN_CONFIG = ObjectConfig(
    name="frying_pan",
    mesh_path=OBJECTS_DIR / "frying_pan_mhndon.obj",
    default_position=(0.0, 0.0, 0.932),
    default_mass=0.0,  # Fixed
    camera_pos=(0.5, 0.5, 1.2),
    camera_target=(0.0, 0.0, 0.932),
)

COFFEE_TABLE_CONFIG = ObjectConfig(
    name="coffee_table",
    mesh_path=OBJECTS_DIR / "coffee_table_koagbh.obj",
    default_position=(0.0, 0.0, 0.0),
    default_mass=0.0,  # Fixed
    camera_pos=(1.0, 1.0, 1.0),
    camera_target=(0.0, 0.0, 0.4),
)


# Object registry
OBJECT_CONFIGS = {
    "fridge": FRIDGE_CONFIG,
    "radio": RADIO_CONFIG,
    "tray": TRAY_CONFIG,
    "frying_pan": FRYING_PAN_CONFIG,
    "coffee_table": COFFEE_TABLE_CONFIG,
}


def get_object_config(object_name: str) -> ObjectConfig:
    """
    Get configuration for an object.
    
    Args:
        object_name: Name of the object
    
    Returns:
        ObjectConfig instance
    
    Raises:
        ValueError: If object name is unknown
    """
    if object_name not in OBJECT_CONFIGS:
        available = ", ".join(OBJECT_CONFIGS.keys())
        raise ValueError(
            f"Unknown object: {object_name}. "
            f"Available objects: {available}"
        )
    
    return OBJECT_CONFIGS[object_name]

