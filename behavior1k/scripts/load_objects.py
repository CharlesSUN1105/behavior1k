#!/usr/bin/env python3
"""
Utility helpers for loading scene objects.

Provides convenience functions to spawn different meshes (table, radio,
fridge, tray, frying pan, etc.) into the PyBullet environment.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pybullet as p

# Ensure the b1k package is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from b1k.paths import models_dir
from pybullet_utils.bullet_client import BulletClient

# ============================================================================
# Constants
# ============================================================================

OBJECTS_DIR = models_dir / "objects"
COFFEE_TABLE_OBJ = OBJECTS_DIR / "coffee_table_koagbh.obj"
RADIO_OBJ = OBJECTS_DIR / "radio_wxnicr.obj"
FRIDGE_OBJ = OBJECTS_DIR / "fridge_dszchb.obj"
TRAY_OBJ = OBJECTS_DIR / "tray_gsxbym.obj"
FRYING_PAN_OBJ = OBJECTS_DIR / "frying_pan_mhndon.obj"


# ============================================================================
# Dataclass definitions
# ============================================================================


@dataclass
class MeshLoadConfig:
    """Configuration for loading a mesh object."""
    obj_path: Path
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass: float = 1.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    use_concave_collision: bool = False


# ============================================================================
# Generic utilities
# ============================================================================

def load_mesh_body(client: BulletClient, cfg: MeshLoadConfig) -> int:
    """
    Load a mesh from an OBJ file into the PyBullet scene.

    Args:
        client: PyBullet client.
        cfg: Mesh loading configuration.

    Returns:
        The created body ID.
    """
    if not cfg.obj_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {cfg.obj_path}")

    visual_shape = client.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(cfg.obj_path),
        meshScale=cfg.mesh_scale,
        specularColor=[0.4, 0.4, 0.4],
    )

    collision_flags = p.GEOM_FORCE_CONCAVE_TRIMESH if cfg.use_concave_collision else 0
    collision_shape = client.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=str(cfg.obj_path),
        meshScale=cfg.mesh_scale,
        flags=collision_flags,
    )

    body_id = client.createMultiBody(
        baseMass=cfg.mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=cfg.position,
        baseOrientation=cfg.orientation,
    )
    return body_id


def align_body_bottom_z(client: BulletClient, body_id: int, target_bottom_z: float):
    """
    Align the bottom of a body to a specific Z height.

    Args:
        client: PyBullet client.
        body_id: Body ID of the object.
        target_bottom_z: Desired bottom height in meters.
    """
    aabb_min, _ = client.getAABB(body_id)
    shift = target_bottom_z - aabb_min[2]
    if abs(shift) < 1e-6:
        return
    pos, orn = client.getBasePositionAndOrientation(body_id)
    client.resetBasePositionAndOrientation(
        body_id, [pos[0], pos[1], pos[2] + shift], orn
    )


def compute_xy_center(aabb_min, aabb_max) -> Tuple[float, float]:
    """
    Compute the XY center of an AABB bounding box.

    Args:
        aabb_min: AABB minimum point [x, y, z].
        aabb_max: AABB maximum point [x, y, z].

    Returns:
        Tuple containing the XY center (x, y).
    """
    return (
        0.5 * (aabb_min[0] + aabb_max[0]),
        0.5 * (aabb_min[1] + aabb_max[1]),
    )


def place_radio_on_table(
    client: BulletClient, radio_id: int, table_top_z: float, center_xy: Tuple[float, float], margin: float = 0.002
):
    """
    Place the radio on the table surface while keeping a small safety margin.

    Args:
        client: PyBullet client.
        radio_id: Body ID of the radio.
        table_top_z: Table top height in meters.
        center_xy: Table center coordinates (x, y).
        margin: Safety margin in meters (default 0.002).
    """
    aabb_min, _ = client.getAABB(radio_id)
    pos, orn = client.getBasePositionAndOrientation(radio_id)
    desired_bottom_z = table_top_z + margin
    shift = desired_bottom_z - aabb_min[2]
    new_pos = [center_xy[0], center_xy[1], pos[2] + shift]
    client.resetBasePositionAndOrientation(radio_id, new_pos, orn)
    client.resetBaseVelocity(radio_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def recenter_body_com_xy(client: BulletClient, body_id: int, target_xy: Tuple[float, float]):
    """
    Align the XY projection of an object's center of mass to a target location.

    Helps avoid tipping caused by an off-center mass.

    Args:
        client: PyBullet client.
        body_id: Body ID of the object.
        target_xy: Target (x, y) coordinates.
    """
    pos, orn = client.getBasePositionAndOrientation(body_id)
    dyn = client.getDynamicsInfo(body_id, -1)
    com_local = dyn[3]
    com_world, _ = p.multiplyTransforms(pos, orn, com_local, [0, 0, 0, 1])
    offset_x = target_xy[0] - com_world[0]
    offset_y = target_xy[1] - com_world[1]
    if abs(offset_x) < 1e-5 and abs(offset_y) < 1e-5:
        return
    client.resetBasePositionAndOrientation(
        body_id,
        [pos[0] + offset_x, pos[1] + offset_y, pos[2]],
        orn,
    )


# ============================================================================
# Object-specific loaders
# ============================================================================

def load_fridge(
    client: BulletClient,
    fridge_position: Tuple[float, float] = (0.0, 0.0),
    fridge_yaw: float = math.pi,  # default 180 degrees
) -> int:
    """Load the fridge mesh and place it stably on the ground.

    Args:
        client: PyBullet client.
        fridge_position: Fridge XY position in meters (default (0.0, 0.0)).
        fridge_yaw: Fridge yaw angle in radians (default π, i.e. 180°).

    Returns:
        Body ID of the fridge.
    """
    fridge_quat = p.getQuaternionFromEuler([0.0, 0.0, fridge_yaw])
    fridge_cfg = MeshLoadConfig(
        obj_path=FRIDGE_OBJ,
        mass=0.0,  # keep fridge static
        use_concave_collision=True,
        position=(fridge_position[0], fridge_position[1], 0.0),
        orientation=fridge_quat,
    )
    fridge_id = load_mesh_body(client, fridge_cfg)
    align_body_bottom_z(client, fridge_id, target_bottom_z=0.0)
    
    client.changeDynamics(
        fridge_id,
        -1,
        lateralFriction=1.0,
        spinningFriction=0.1,
        rollingFriction=0.0,
        restitution=0.0,
    )
    
    return fridge_id


def load_tray(
    client: BulletClient,
    tray_position: Tuple[float, float] = (0.0, 0.0),
    tray_height: float = 0.8910,
    tray_yaw: float = 0.0,
) -> int:
    """Load the tray mesh and keep it at a fixed position/height.

    Args:
        client: PyBullet client.
        tray_position: Tray XY position in meters (default (0.0, 0.0)).
        tray_height: Tray bottom height in meters (default 0.8910).
        tray_yaw: Tray yaw angle in radians (default 0.0).

    Returns:
        Body ID of the tray.
    """
    tray_quat = p.getQuaternionFromEuler([0.0, 0.0, tray_yaw])
    tray_cfg = MeshLoadConfig(
        obj_path=TRAY_OBJ,
        mass=0.0,  # keep tray static
        use_concave_collision=True,
        position=(tray_position[0], tray_position[1], tray_height),
        orientation=tray_quat,
    )
    tray_id = load_mesh_body(client, tray_cfg)
    
    # Align the tray bottom with the target height
    align_body_bottom_z(client, tray_id, target_bottom_z=tray_height)
    
    client.changeDynamics(
        tray_id,
        -1,
        lateralFriction=1.0,
        spinningFriction=0.1,
        rollingFriction=0.0,
        restitution=0.0,
    )
    
    return tray_id


def load_frying_pan(
    client: BulletClient,
    frying_pan_position: Tuple[float, float] = (0.0, 0.0),
    frying_pan_height: float = 0.89,
    frying_pan_yaw: float = 0.0,
) -> int:
    """Load the frying pan mesh and lock it at a given pose.

    Args:
        client: PyBullet client.
        frying_pan_position: Frying pan XY position in meters (default (0.0, 0.0)).
        frying_pan_height: Frying pan bottom height in meters (default 0.89).
        frying_pan_yaw: Frying pan yaw angle in radians (default 0.0).

    Returns:
        Body ID of the frying pan.
    """
    frying_pan_quat = p.getQuaternionFromEuler([0.0, 0.0, frying_pan_yaw])
    frying_pan_cfg = MeshLoadConfig(
        obj_path=FRYING_PAN_OBJ,
        mass=0.0,  # keep frying pan static
        use_concave_collision=True,
        position=(frying_pan_position[0], frying_pan_position[1], frying_pan_height),
        orientation=frying_pan_quat,
    )
    frying_pan_id = load_mesh_body(client, frying_pan_cfg)
    
    # Align the frying pan bottom with the target height
    align_body_bottom_z(client, frying_pan_id, target_bottom_z=frying_pan_height)
    
    client.changeDynamics(
        frying_pan_id,
        -1,
        lateralFriction=1.0,
        spinningFriction=0.1,
        rollingFriction=0.0,
        restitution=0.0,
    )
    
    return frying_pan_id


# ============================================================================
# Composite loaders
# ============================================================================

def load_table_and_radio(
    client: BulletClient,
    table_xy: Tuple[float, float] = (0.0, 0.0),
    table_yaw: float = 0.0,
    fix_radio: bool = False,
) -> Tuple[int, int]:
    """
    Load a table and a radio, placing the radio safely on top of the table.

    Args:
        client: PyBullet client.
        table_xy: Table center XY position (default (0.0, 0.0)).
        table_yaw: Table yaw in radians (default 0.0).
        fix_radio: Whether to keep the radio static (mass = 0), default False.

    Returns:
        Tuple of (table_id, radio_id).
    """
    # Load table
    table_quat = p.getQuaternionFromEuler([0.0, 0.0, table_yaw])
    table_cfg = MeshLoadConfig(
        obj_path=COFFEE_TABLE_OBJ,
        mass=0.0,
        use_concave_collision=True,
        position=(table_xy[0], table_xy[1], 0.0),
        orientation=table_quat,
    )
    table_id = load_mesh_body(client, table_cfg)
    align_body_bottom_z(client, table_id, target_bottom_z=0.0)
    
    # Retrieve table bounds
    table_aabb_min, table_aabb_max = client.getAABB(table_id)
    table_center_xy = compute_xy_center(table_aabb_min, table_aabb_max)
    table_top_z = table_aabb_max[2]
    
    # Configure table dynamics
    client.changeDynamics(
        table_id,
        -1,
        lateralFriction=1.0,
        spinningFriction=0.1,
        rollingFriction=0.0,
        restitution=0.0,
    )
    
    # Load radio
    radio_quat = p.getQuaternionFromEuler([0.0, 0.0, math.radians(90.0)])  # rotate 90° around Z
    radio_cfg = MeshLoadConfig(
        obj_path=RADIO_OBJ,
        mass=0.0 if fix_radio else 2.0,
        position=(table_center_xy[0], table_center_xy[1], table_top_z + 0.5),
        orientation=radio_quat,
    )
    radio_id = load_mesh_body(client, radio_cfg)
    place_radio_on_table(client, radio_id, table_top_z, table_center_xy)
    recenter_body_com_xy(client, radio_id, table_center_xy)
    
    # Configure radio dynamics
    client.changeDynamics(
        radio_id,
        -1,
        lateralFriction=1.2,
        spinningFriction=0.2,
        rollingFriction=0.0001,
        restitution=0.0,
        contactDamping=0.8,
        contactStiffness=10_000.0,
        angularDamping=0.99,
        linearDamping=0.2,
    )
    
    if fix_radio:
        client.resetBaseVelocity(radio_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    return table_id, radio_id
