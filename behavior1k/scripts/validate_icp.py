#!/usr/bin/env python3
"""ICP validation helper.

Captures RGB/Depth/Segmentation for a selected object, runs the ICP pipeline,
and writes every intermediate artifact to disk so that issues can be diagnosed
quickly (mask coverage, point cloud stats, fitness score, etc.).
python behavior1k/scripts/validate_icp.py --object fridge --use-point-to-plane --use-coarse-alignment 
--use-dual-cameras

python - <<'PY'
import numpy as np, open3d as o3d
base = "/Users/charles/Downloads/bullet3/icp_validation/fridge_20251110_124325"

obs = o3d.io.read_point_cloud(f"{base}/observed_processed.ply")
model = o3d.io.read_point_cloud(f"{base}/model_processed.ply")
pose = np.load(f"{base}/pose_camera.npy")

obs.transform(np.linalg.inv(pose)) 
obs.paint_uniform_color([1, 0, 0])
model.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([obs, model])
PY

python - <<'PY'
import numpy as np
import open3d as o3d

base="icp_validation/fridge_20251110_102003"
pcd=o3d.io.read_point_cloud(f"{base}/observed_processed.ply")
pose=np.load(f"{base}/pose_robot_base.npy")

print("pose_robot_base:\n", pose)
pcd.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([pcd])
PY
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

# Add repository root to path (same as other scripts)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from b1k.vision.camera import RGBDCamera
from b1k.vision.segmentation import ObjectSegmentor, capture_segmentation_image
from b1k.vision.pointcloud import PointCloudProcessor, load_mesh_as_pointcloud
from b1k.vision.icp import ICPMatcher, evaluate_registration, coarse_alignment_ransac
from b1k.vision.transforms import CoordinateTransformer, pose_to_matrix
from b1k.pybullet_utils import init_client
from demo_icp import (
    OBJECT_CONFIGS,
    CAMERA_LINK_DEFAULTS,
    DEFAULT_CAMERA_POS,
    DEFAULT_CAMERA_TARGET,
    DEFAULT_SECOND_CAMERA_POS,
    DEFAULT_SECOND_CAMERA_TARGET,
    DEFAULT_SECOND_CAMERA_UP,
)
import load_objects
import load_r1pro

try:
    from PIL import Image

    def _save_image(data: np.ndarray, path: Path) -> None:
        Image.fromarray(data).save(str(path))


    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path

    def _save_image(data: np.ndarray, path: Path) -> None:
        np.save(path.with_suffix(path.suffix + ".npy"), data)


PIL_AVAILABLE = False


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Camera axis vector cannot be zero")
    return vec / norm


def get_camera_axes(
    link_name: str,
    user_forward: Optional[Tuple[float, float, float]],
    user_up: Optional[Tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    defaults = CAMERA_LINK_DEFAULTS.get(link_name, {})
    forward = (
        np.array(user_forward, dtype=float)
        if user_forward is not None
        else defaults.get("forward", np.array([1.0, 0.0, 0.0]))
    )
    up = (
        np.array(user_up, dtype=float)
        if user_up is not None
        else defaults.get("up", np.array([0.0, 0.0, 1.0]))
    )
    return forward, up


def compute_camera_view_from_link(
    client,
    robot_id: int,
    link_name: str,
    forward_axis: np.ndarray,
    up_axis: np.ndarray,
    debug: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    """Return camera position/target/up vectors from link pose."""
    link_index = -1
    base_name = client.getBodyInfo(robot_id)[0].decode("utf-8")
    if base_name != link_name:
        num_joints = client.getNumJoints(robot_id)
        for joint_index in range(num_joints):
            joint_info = client.getJointInfo(robot_id, joint_index)
            if joint_info[12].decode("utf-8") == link_name:
                link_index = joint_index
                break
    if base_name != link_name and link_index == -1:
        raise ValueError(f"Link '{link_name}' not found")

    if base_name == link_name:
        position, orientation = client.getBasePositionAndOrientation(robot_id)
    else:
        link_state = client.getLinkState(
            robot_id, link_index, computeForwardKinematics=True
        )
        position = link_state[0]
        orientation = link_state[1]

    rot = np.array(client.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    forward_world = rot @ _normalize(forward_axis)
    up_world = rot @ _normalize(up_axis)
    camera_pos = np.array(position)
    camera_target = camera_pos + forward_world
    if debug:
        scale = 0.3
        client.addUserDebugLine(
            camera_pos,
            camera_pos + forward_world * scale,
            lineColorRGB=[1, 0, 0],
            lifeTime=1.0,
        )
        client.addUserDebugLine(
            camera_pos,
            camera_pos + up_world * scale,
            lineColorRGB=[0, 1, 0],
            lifeTime=1.0,
        )
    return camera_pos.tolist(), camera_target.tolist(), up_world.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object",
        required=True,
        choices=list(OBJECT_CONFIGS.keys()),
        help="Object to validate (same list as demo_icp).",
    )
    parser.add_argument(
        "--camera-link",
        default=None,
        help="Optional camera link to align with (default: use preset camera pose).",
    )
    parser.add_argument(
        "--free-camera",
        action="store_true",
        help="Use preset/overridden camera pose instead of camera link pose.",
    )
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override camera position (world frame).",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override camera target (world frame).",
    )
    parser.add_argument(
        "--camera-forward-axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Camera forward axis in link frame (defaults to per-link config).",
    )
    parser.add_argument(
        "--camera-up-axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Camera up axis in link frame or free camera up vector.",
    )
    parser.add_argument(
        "--visualize-camera-axes",
        action="store_true",
        help="Draw debug lines for camera forward/up directions.",
    )
    parser.add_argument(
        "--use-sam",
        action="store_true",
        help="Use SAM for segmentation instead of PyBullet IDs.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (required when --use-sam).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for point-cloud preprocessing (meters).",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=0.05,
        help="ICP max correspondence distance (meters).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum ICP iterations.",
    )
    parser.add_argument(
        "--use-point-to-plane",
        action="store_true",
        help="Use point-to-plane ICP (requires normals).",
    )
    parser.add_argument(
        "--use-dual-cameras",
        action="store_true",
        help="Capture a second camera view (free camera mode only).",
    )
    parser.add_argument(
        "--second-camera-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override second camera position when --use-dual-cameras is set.",
    )
    parser.add_argument(
        "--second-camera-target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override second camera target when --use-dual-cameras is set.",
    )
    parser.add_argument(
        "--second-camera-up",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override second camera up vector when --use-dual-cameras is set.",
    )
    parser.add_argument(
        "--use-coarse-alignment",
        action="store_true",
        help="Run FPFH+RANSAC coarse alignment before ICP refinement.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where all artifacts will be saved.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run PyBullet in DIRECT mode (no GUI).",
    )
    parser.add_argument(
        "--object-position",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Override default object XY placement (meters).",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep simulation window open after finishing (GUI mode only).",
    )
    parser.add_argument(
        "--save-camera-pose",
        action="store_true",
        help="After validation, prompt to save the current GUI camera pose to output directory.",
    )
    parser.add_argument(
        "--camera-pose-filename",
        type=str,
        default="camera_pose.json",
        help="Filename for storing camera pose JSON when --save-camera-pose is set.",
    )
    return parser.parse_args()


def ensure_output_dir(base_dir: Optional[str], object_name: str) -> Path:
    if base_dir:
        out = Path(base_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "icp_validation" / f"{object_name}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_depth_image(depth: np.ndarray, path: Path) -> None:
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    span = depth_max - depth_min
    if span <= 1e-6:
        span = 1.0
    normalized = (depth - depth_min) / span
    depth_uint16 = (normalized * 65535).astype(np.uint16)
    _save_image(depth_uint16, path)


def place_object(obj_name: str, client, override_xy: Optional[list[float]]):
    cfg = OBJECT_CONFIGS[obj_name]
    if override_xy is None:
        return cfg["load_func"](client)

    if obj_name == "fridge":
        return cfg["load_func"](client, fridge_position=tuple(override_xy))
    if obj_name == "tray":
        return cfg["load_func"](client, tray_position=tuple(override_xy))
    if obj_name == "frying_pan":
        return cfg["load_func"](
            client, frying_pan_position=tuple(override_xy)
        )
    return cfg["load_func"](client)


def build_initial_transform(observed_pcd, model_pcd) -> np.ndarray:
    transform = np.eye(4)
    observed_center = np.asarray(observed_pcd.get_center())
    model_center = np.asarray(model_pcd.get_center())
    # Move observed cloud towards the model so centroids roughly coincide.
    # registration_icp expects a transform that maps observed -> model, so we
    # need target - source rather than the opposite.
    transform[:3, 3] = model_center - observed_center
    return transform


def save_gui_camera_pose(client, output_dir: Path, filename: str) -> Path:
    info = client.getDebugVisualizerCamera()
    camera_pos = info[11]
    forward = info[5]
    distance = info[8]
    camera_target = [camera_pos[i] + forward[i] * distance for i in range(3)]
    camera_up = info[13]

    data = {
        "camera_pos": camera_pos,
        "camera_target": camera_target,
        "camera_up": camera_up,
    }

    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir, args.object)
    print(f"Artifacts will be saved to: {output_dir}")

    client = init_client(gui=not args.headless)
    client.setGravity(0, 0, -9.81)
    client.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=40,
        cameraPitch=-30,
        cameraTargetPosition=[0.0, 0.0, 0.5],
    )
    client.loadURDF("plane.urdf")

    robot_id = load_r1pro.load_r1pro(client, base_position=[-1.0, 0.0, 0.0])

    print(f"Loading {args.object}...")
    object_id = place_object(args.object, client, args.object_position)
    for _ in range(240):
        client.stepSimulation()
    print(f"✓ Object loaded with body_id={object_id}")

    cfg = OBJECT_CONFIGS[args.object]
    forward_axis, up_axis = get_camera_axes(
        args.camera_link,
        tuple(args.camera_forward_axis) if args.camera_forward_axis else None,
        tuple(args.camera_up_axis) if args.camera_up_axis else None,
    )
    use_link_camera = args.camera_link is not None and not args.free_camera

    if not use_link_camera:
        default_pos = list(DEFAULT_CAMERA_POS)
        default_target = list(DEFAULT_CAMERA_TARGET)
        camera_pos = args.camera_pos or default_pos
        camera_target = args.camera_target or default_target
        if args.camera_up_axis:
            camera_up = tuple(_normalize(np.array(args.camera_up_axis)).tolist())
        else:
            camera_up = (0.0, 0.0, 1.0)
    else:
        camera_pos, camera_target, _ = compute_camera_view_from_link(
            client,
            robot_id,
            args.camera_link,
            forward_axis,
            up_axis,
            debug=args.visualize_camera_axes,
        )
        if args.camera_up_axis:
            camera_up = tuple(_normalize(np.array(args.camera_up_axis)).tolist())
        else:
            camera_up = (0.0, 0.0, 1.0)
        print(
            f"Using camera link '{args.camera_link}' @ pos {camera_pos} looking {camera_target}"
        )

    second_pos_default = tuple(cfg.get("second_camera_pos", DEFAULT_SECOND_CAMERA_POS))
    second_target_default = tuple(
        cfg.get("second_camera_target", DEFAULT_SECOND_CAMERA_TARGET)
    )
    second_up_default = tuple(
        _normalize(np.array(cfg.get("second_camera_up", DEFAULT_SECOND_CAMERA_UP)))
    )

    transformer = CoordinateTransformer(client, robot_id)
    camera = RGBDCamera(client)
    intrinsic = camera.get_intrinsic_matrix()
    segmentor = ObjectSegmentor(args.use_sam, args.sam_checkpoint)
    point_processor = PointCloudProcessor(voxel_size=args.voxel_size)

    camera_setups = [
        {
            "name": "cam0",
            "pos": tuple(camera_pos),
            "target": tuple(camera_target),
            "up": tuple(camera_up),
        }
    ]
    if args.use_dual_cameras:
        if use_link_camera:
            raise ValueError("--use-dual-cameras 仅支持自由相机模式（--free-camera）。")
        mirrored_pos = (
            tuple(args.second_camera_pos)
            if args.second_camera_pos is not None
            else second_pos_default
        )
        mirrored_target = (
            tuple(args.second_camera_target)
            if args.second_camera_target is not None
            else second_target_default
        )
        if args.second_camera_up is not None:
            second_up = tuple(_normalize(np.array(args.second_camera_up)).tolist())
        else:
            second_up = second_up_default
        camera_setups.append(
            {
                "name": "cam1",
                "pos": mirrored_pos,
                "target": mirrored_target,
                "up": second_up,
            }
        )

    camera_pose_worlds: list[np.ndarray] = []
    camera_to_base_list: list[np.ndarray] = []
    camera_raw_clouds: list[o3d.geometry.PointCloud] = []
    camera_processed_clouds: list[o3d.geometry.PointCloud] = []
    total_mask_pixels = 0
    total_raw_points = 0
    total_processed_points = 0
    per_camera_stats: list[str] = []

    base_position, base_orientation = client.getBasePositionAndOrientation(robot_id)
    base_in_world = pose_to_matrix(base_position, base_orientation)
    world_to_base = np.linalg.inv(base_in_world)

    for idx, cam_cfg in enumerate(camera_setups):
        print(
            f"Capturing RGBD for {cam_cfg['name']} @ pos {cam_cfg['pos']} looking {cam_cfg['target']}"
        )
        rgb, depth, view_matrix, projection_matrix = camera.capture(
            cam_cfg["pos"], cam_cfg["target"], cam_cfg["up"]
        )
        camera_pose_world = np.linalg.inv(view_matrix)
        camera_pose_worlds.append(camera_pose_world)

        np.save(output_dir / f"depth_cam{idx}.npy", depth)
        np.save(output_dir / f"view_matrix_cam{idx}.npy", view_matrix)
        np.save(output_dir / f"projection_matrix_cam{idx}.npy", projection_matrix)
        np.save(output_dir / f"camera_pose_world_cam{idx}.npy", camera_pose_world)
        _save_image(rgb, output_dir / f"rgb_cam{idx}.png")
        save_depth_image(depth, output_dir / f"depth_cam{idx}.png")

        if idx == 0:
            np.save(output_dir / "depth.npy", depth)
            np.save(output_dir / "view_matrix.npy", view_matrix)
            np.save(output_dir / "projection_matrix.npy", projection_matrix)
            np.save(output_dir / "camera_pose_world.npy", camera_pose_world)
            _save_image(rgb, output_dir / "rgb.png")
            save_depth_image(depth, output_dir / "depth.png")

        seg_image = capture_segmentation_image(
            client,
            cam_cfg["pos"],
            cam_cfg["target"],
            camera_up=cam_cfg["up"],
            width=camera.width,
            height=camera.height,
        )
        np.save(output_dir / f"segmentation_cam{idx}.npy", seg_image)
        if idx == 0:
            np.save(output_dir / "segmentation.npy", seg_image)

        if args.use_sam:
            mask = segmentor.segment(rgb, method="sam")
        else:
            mask = segmentor.segment(
                rgb,
                method="pybullet",
                object_id=object_id,
                seg_image=seg_image,
            )
        mask_uint8 = (mask.astype(np.uint8) * 255)
        _save_image(mask_uint8, output_dir / f"mask_cam{idx}.png")
        np.save(output_dir / f"mask_cam{idx}.npy", mask)
        if idx == 0:
            _save_image(mask_uint8, output_dir / "mask.png")
            np.save(output_dir / "mask.npy", mask)

        mask_pixels = int(mask.sum())
        total_mask_pixels += mask_pixels

        observed_raw = point_processor.extract_from_rgbd(rgb, depth, mask, intrinsic)
        raw_points = len(observed_raw.points)
        total_raw_points += raw_points
        o3d.io.write_point_cloud(
            str(output_dir / f"observed_raw_cam{idx}.ply"), observed_raw
        )

        observed_processed = point_processor.preprocess(observed_raw)
        processed_points = len(observed_processed.points)
        total_processed_points += processed_points
        if observed_processed.is_empty():
            raise ValueError(
                f"{cam_cfg['name']} 的点云在预处理后为空，请检查遮罩。"
            )
        o3d.io.write_point_cloud(
            str(output_dir / f"observed_processed_cam{idx}.ply"), observed_processed
        )

        camera_raw_clouds.append(o3d.geometry.PointCloud(observed_raw))
        camera_processed_clouds.append(o3d.geometry.PointCloud(observed_processed))

        per_camera_stats.append(
            f"cam{idx}: mask_pixels={mask_pixels}, raw_points={raw_points}, processed_points={processed_points}"
        )
        print(
            f"{cam_cfg['name']} point count (raw -> processed): {raw_points} -> {processed_points}"
        )

        if use_link_camera:
            camera_to_base = transformer.get_camera_to_base_transform(args.camera_link)
        else:
            camera_to_base = world_to_base @ camera_pose_world
        camera_to_base_list.append(camera_to_base)

    mask_pixels = total_mask_pixels
    print(
        f"Mask pixels (total): {total_mask_pixels} across {len(camera_setups)} camera(s)"
    )

    np.save(output_dir / "camera_pose_world_all.npy", np.stack(camera_pose_worlds))
    camera_pose_world = camera_pose_worlds[0]
    np.save(output_dir / "intrinsics.npy", intrinsic)
    if not camera_processed_clouds:
        raise ValueError("未能生成有效的融合点云。")

    combined_raw = o3d.geometry.PointCloud(camera_raw_clouds[0])
    combined_processed = o3d.geometry.PointCloud(camera_processed_clouds[0])

    if len(camera_processed_clouds) > 1:
        print("Aligning secondary camera point clouds to primary view via ICP...")
        align_matcher = ICPMatcher(
            max_correspondence_distance=max(
                0.02, args.max_correspondence_distance * 0.6
            ),
            max_iterations=max(30, args.max_iterations // 2),
            use_point_to_plane=True,
        )
        reference_pose = camera_pose_worlds[0]
        reference_cloud = o3d.geometry.PointCloud(combined_processed)

        for cam_idx in range(1, len(camera_processed_clouds)):
            src_cloud = camera_processed_clouds[cam_idx]
            if src_cloud.is_empty():
                continue
            init_transform = np.linalg.inv(reference_pose) @ camera_pose_worlds[cam_idx]
            transform, fitness_local = align_matcher.estimate_pose(
                o3d.geometry.PointCloud(src_cloud),
                reference_cloud,
                initial_transform=init_transform,
            )
            print(
                f"  cam{cam_idx} alignment fitness={fitness_local:.3f} (initial guess from extrinsics)"
            )
            aligned_processed = o3d.geometry.PointCloud(src_cloud)
            aligned_processed.transform(transform)
            combined_processed += aligned_processed

            aligned_raw = o3d.geometry.PointCloud(camera_raw_clouds[cam_idx])
            aligned_raw.transform(transform)
            combined_raw += aligned_raw

            reference_cloud = o3d.geometry.PointCloud(combined_processed)

    o3d.io.write_point_cloud(str(output_dir / "observed_raw.ply"), combined_raw)
    o3d.io.write_point_cloud(
        str(output_dir / "observed_processed.ply"), combined_processed
    )
    np.save(
        output_dir / "observed_raw_point_count.npy", np.array(total_raw_points)
    )
    observed = combined_processed
    print(
        f"Combined observed point count (raw -> processed): {len(combined_raw.points)} -> {len(observed.points)}"
    )

    model_pcd = load_mesh_as_pointcloud(str(cfg["mesh_path"]))
    model_pcd = point_processor.preprocess(model_pcd)
    o3d.io.write_point_cloud(
        str(output_dir / "model_processed.ply"), model_pcd
    )

    icp = ICPMatcher(
        max_correspondence_distance=args.max_correspondence_distance,
        max_iterations=args.max_iterations,
        use_point_to_plane=args.use_point_to_plane,
    )

    if args.use_coarse_alignment:
        print("Running coarse alignment (FPFH + RANSAC) to initialize ICP...")
        initial_transform = coarse_alignment_ransac(observed, model_pcd)
    else:
        initial_transform = build_initial_transform(observed, model_pcd)
    np.save(output_dir / "initial_transform.npy", initial_transform)

    icp_transform, fitness = icp.estimate_pose(
        observed,
        model_pcd,
        initial_transform=initial_transform,
    )
    pose_camera = np.linalg.inv(icp_transform)
    np.save(output_dir / "pose_camera.npy", pose_camera)
    print(f"ICP fitness: {fitness:.3f}")
    print(f"Camera-frame translation: {pose_camera[:3, 3]}")

    pose_base = camera_to_base_list[0] @ pose_camera
    np.save(output_dir / "pose_robot_base.npy", pose_base)
    print(f"Detected base-frame position: {pose_base[:3, 3]}")

    fitness_eval, rmse = evaluate_registration(
        observed,
        model_pcd,
        icp_transform,
        max_correspondence_distance=args.max_correspondence_distance,
    )
    print(f"Evaluation fitness: {fitness_eval:.3f}, RMSE: {rmse:.4f}")
    summary_lines = [
        f"num_cameras={len(camera_setups)}",
        f"mask_pixels_total={mask_pixels}",
        f"observed_points_raw_total={total_raw_points}",
        f"observed_points_processed_sum={total_processed_points}",
        f"observed_points_processed_combined={len(observed.points)}",
        f"fitness={fitness}",
        f"eval_fitness={fitness_eval}",
        f"rmse={rmse}",
    ]
    summary_lines.extend(per_camera_stats)
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))

    if args.save_camera_pose:
        if args.headless:
            print("⚠️ --save-camera-pose ignored in headless mode (no GUI available).")
        else:
            input(
                "\n📸 调整好 GUI 中的摄像机后按 Enter 保存当前位姿... "
            )
            pose_file = save_gui_camera_pose(
                client, output_dir, args.camera_pose_filename
            )
            print(f"✓ 相机位姿已保存到 {pose_file}")

    if not args.headless and args.keep_running:
        print("Validation complete. Press Ctrl+C in the console to exit.")
        try:
            while True:
                client.stepSimulation()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"❌ Validation failed: {exc}")
        sys.exit(1)
