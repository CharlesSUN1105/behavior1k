# New Additions Overview

This document summarizes the recently introduced modules and scripts so that you can
quickly understand what each piece does and how to use it in daily workflows.


## Object Loading Utilities (`scripts/load_objects.py`)

- Provides typed helpers (`MeshLoadConfig`) and a set of convenience functions for spawning
  common meshes (table, radio, fridge, tray, frying pan) in PyBullet.
- Every helper automatically:
  - Loads the OBJ mesh with visual + collision shapes.
  - Aligns the bottom surface to the requested height.
  - Applies reasonable dynamics parameters (friction, damping, etc.).
- Key entry points:
  - `load_table_and_radio(client, table_xy, table_yaw, fix_radio)`
  - `load_fridge(client, fridge_position, fridge_yaw)`
  - `load_tray(client, tray_position, tray_height, tray_yaw)`
  - `load_frying_pan(client, frying_pan_position, frying_pan_height, frying_pan_yaw)`
- All docstrings, comments, and error messages are now in English and include parameter
  descriptions so IDEs and auto-complete remain friendly.


## Robot Loader and Interactive Control (`scripts/load_r1pro.py`)

- Adds an English-only command help to load the R1Pro robot together with optional
  objects using CLI flags (table, radio, fridge, tray, frying pan).
- Supports applying any predefined grasp pose on startup via `--grasp-pose`.
- Interactive mode:
  - Launch with `--interactive` to receive both GUI sliders and terminal overrides.
  - Manual commands (`base_x`, `base_y`, `joint`, `reset`, `help`, `quit`) are printed in English.
  - Messages clearly describe fallback behaviour when GUI sliders are unavailable.
- Stabilization utilities:
  - `stabilize_scene` steps the simulation to settle physics before user interaction.


## Vision & ICP Modules (`b1k/vision/`)

These modules implement the vision-based grasp adjustment pipeline.

- `camera.py`: camera utilities (RGB, depth capture, intrinsic/extrinsic handling).
- `segmentation.py`: segmentation helpers for mask generation (PyBullet mask or external tools).
- `pointcloud.py`: RGB-D → point cloud conversion, filtering, and transformations.
- `icp.py`: Open3D-based ICP routines (classic point-to-plane method).
- `transforms.py`: coordinate transforms between camera, object, and robot frames.
- `pose_adjuster.py`: orchestrates the entire pipeline—captures data, runs segmentation,
  performs ICP, and outputs an adjusted grasp pose.
- `__init__.py`: exposes key entry points for downstream imports.

> Tip: All modules avoid direct Open3D imports at runtime unless needed (using `TYPE_CHECKING`)
> so that type hints work even when Open3D is optional.


## Demo and Validation Scripts

### `scripts/demo_icp.py`
- Demonstrates how to apply the vision pipeline for different objects.
- Supports:
  - Loading a target object with consistent robot poses.
  - Using predefined camera poses or deriving them from robot link frames (`zed_link`, etc.).
  - Multi-camera configurations through `OBJECT_CONFIGS` (primary + secondary view).
- Command example:
  ```bash
  python behavior1k/scripts/demo_icp.py \
    --object fridge \
    --camera-link zed_link \
    --grasp-pose open_fridge_right
  ```

### `scripts/validate_icp.py`
- Runs the full ICP pipeline with optional file exports (RGB, depth, masks, point clouds).
- Includes the ability to:
  - Capture GUI camera poses via `--save-camera-pose`.
  - Force preset cameras or use robot-mounted links.
  - Save intermediate artifacts for debugging (`observed_processed.ply`, `model_processed.ply`).
- Command example:
  ```bash
  python behavior1k/scripts/validate_icp.py \
    --object radio \
    --camera-link zed_link \
    --output-dir icp_validation/radio_test
  ```


## Recommended Workflow

1. **Quick inspection**  
   - `python scripts/load_r1pro.py --with-objects --interactive`
   - Adjust robot pose, export joints via `show_joints` in the terminal.

2. **Vision-based pose adjustment**  
   - Use `demo_icp.py` for a guided run with one object.
   - Switch to `validate_icp.py` when you need artifacts for debugging or reporting fitness/RMSE.

3. **Adding new objects or poses**  
   - Extend `scripts/load_objects.py` with a new helper.
   - Update `OBJECT_CONFIGS` and camera presets in `demo_icp.py` / `validate_icp.py`.
   - Add new grasp poses to `b1k/models/r1pro/grasp_poses.py` and JSON counterpart.


## Environment Notes

- The expected conda environment name is `grasp_pose` (as provided earlier).
- Ensure the environment includes:
  - `pybullet`
  - `open3d`
  - `numpy`
  - Any segmentation dependencies (e.g. SAM if used externally).
- When using the GUI alongside scripts, avoid killing the PyBullet window abruptly to keep shared memory clean.


## Troubleshooting

- **Robot loads above the ground**  
  Check that the chosen grasp pose sets base Z to 0.0. `apply_grasp_pose` enforces this.

- **Sliders missing**  
  Ensure the GUI is enabled (`init_client(gui=True)`) and no previous PyBullet session is holding shared memory.

- **ICP returns fitness 0**  
  Usually indicates the initial alignment is too poor—try a better camera view or adjust `max_correspondence_distance` in `validate_icp.py`.

- **Camera pose feels tilted**  
  Modify `CAMERA_LINK_DEFAULTS` in `demo_icp.py` or capture a preset via `--save-camera-pose` in `validate_icp.py`.


## Where to Go Next

- Extend the pose library with additional task-specific configurations.
- Integrate the vision pipeline (`pose_adjuster.py`) into higher-level task planners.
- Experiment with multi-camera setups by duplicating the camera capture stages in `validate_icp.py`.


Happy experimenting! Let us know if additional documentation would be helpful for governance,
testing, or deployment workflows.

