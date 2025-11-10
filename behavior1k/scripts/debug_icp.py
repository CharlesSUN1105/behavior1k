#!/usr/bin/env python3
"""Visualize ICP alignment in both camera and base frames."""

import numpy as np
import open3d as o3d

BASE = "icp_validation/fridge_20251110_102003"

def main() -> None:
    base = BASE

    pose_camera = np.load(f"{base}/pose_camera.npy")
    pose_base = np.load(f"{base}/pose_robot_base.npy")

    print("pose_camera translation:", pose_camera[:3, 3])
    print("pose_robot_base translation:", pose_base[:3, 3])

    observed = o3d.io.read_point_cloud(f"{base}/observed_processed.ply")
    model_pcd = o3d.io.read_point_cloud(f"{base}/model_processed.ply")
    print("Model point count:", len(model_pcd.points))

    observed_in_model = o3d.geometry.PointCloud(observed)
    observed_in_model.transform(pose_camera)
    observed_in_model.paint_uniform_color([1, 0, 0])

    model_in_model = o3d.geometry.PointCloud(model_pcd)
    model_in_model.paint_uniform_color([0, 1, 0])

    model_in_base = o3d.geometry.PointCloud(model_pcd)
    model_in_base.transform(pose_base)
    model_in_base.paint_uniform_color([0, 1, 0])

    observed_in_base = o3d.geometry.PointCloud(observed)
    observed_in_base.transform(pose_camera)
    observed_in_base.transform(pose_base)
    observed_in_base.paint_uniform_color([1, 0, 0])

    print("Showing camera-frame alignment (model coords)...")
    o3d.visualization.draw_geometries([model_in_model, observed_in_model])
    print("Showing base-frame alignment...")
    o3d.visualization.draw_geometries([model_in_base, observed_in_base])

    residual = pose_base @ np.linalg.inv(pose_camera) @ pose_camera - pose_base
    print("Chain residual (should be near zero):\n", residual)


if __name__ == "__main__":
    main()
