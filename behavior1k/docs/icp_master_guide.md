# ICP Master Guide

> 一份整合所有 ICP 相关文档（Quickstart、README、Summary、Quick Reference、Usage Guide、Code Organization、Vision Based Grasp Adjustment）的总览手册。

---

## 1. 快速上手

### 1.1 环境验证

```bash
# 1. 验证模块导入
python -c "from b1k.vision import GraspPoseAdjuster; print('✓ OK')"

# 2. 检查核心目录
ls -lh b1k/vision/
```

### 1.2 三行代码体验

```python
from b1k.vision import GraspPoseAdjuster

adjuster = GraspPoseAdjuster(client, robot_id, "b1k/models/objects/fridge_dszchb.obj", "zed_link")
pose, fitness = adjuster.detect_object_pose(camera_pos, camera_target, object_id)
adjusted = adjuster.adjust_grasp_pose(base_pose, pose)
```

### 1.3 推荐学习路线

1. 运行演示脚本：`python scripts/demo_icp.py --object fridge`
2. 查阅速查表：`docs/icp_quick_reference.md`
3. 阅读 API 指南：`docs/icp_usage_guide.md`
4. 根据本手册第 3 节集成到业务代码

---

## 2. 系统概览

### 2.1 目录结构

```
behavior1k/
├── b1k/vision/                 # ICP 核心模块（7个文件）
│   ├── camera.py               # RGBD 相机
│   ├── segmentation.py         # 物体分割（PyBullet / SAM）
│   ├── pointcloud.py           # 点云处理
│   ├── icp.py                  # ICP 匹配与粗对齐
│   ├── transforms.py           # 坐标转换
│   └── pose_adjuster.py        # 主入口（GraspPoseAdjuster）
├── scripts/demo_icp.py         # 演示脚本
├── b1k/models/objects/object_configs.py  # 物体配置
└── docs/…                     # 其余文档（已整合于本手册）
```

### 2.2 核心能力

| 模块 | 职责 | 说明 |
|------|------|------|
| camera.py | 获取 RGBD 数据 | 使用 PyBullet 虚拟相机，支持自定义参数 |
| segmentation.py | SAM / PyBullet 分割 | 默认 PyBullet 分割，需物体 ID |
| pointcloud.py | 提取/滤波点云 | 支持降采样、去噪、法线估计 |
| icp.py | ICP 姿态估计 | 提供粗对齐 + 点到点/点到面 ICP |
| transforms.py | 坐标转换 | 相机系 → 机器人基座系 |
| pose_adjuster.py | 一站式调用 | `GraspPoseAdjuster` 主入口 |

### 2.3 默认相机

- 默认使用头部相机 `zed_link`。
- 其他可选：`right_realsense_link`（右臂）、`left_realsense_link`（左臂）。
- 运行 `python scripts/demo_icp.py --camera-link <link>` 可覆盖默认值。

---

## 3. 使用方式

### 3.1 演示脚本

```bash
# 自动检测并调整
python scripts/demo_icp.py --object fridge --grasp-pose open_fridge_right

# 指定其它物体
python scripts/demo_icp.py --object radio
python scripts/demo_icp.py --object tray
python scripts/demo_icp.py --object frying_pan

# 使用 SAM 分割
python scripts/demo_icp.py --object fridge --use-sam --sam-checkpoint sam.pth

# 固定姿态（跳过 ICP）
python scripts/demo_icp.py --object fridge --no-auto-detect
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--camera-link` | 选择相机 link（默认 `zed_link`） |
| `--use-sam` | 切换到 SAM 分割 |
| `--no-auto-detect` | 不执行 ICP，直接使用预设抓取姿态 |
| `--object-position` / `--fridge-position` | 覆盖默认物体位置 |

### 3.2 Python API（推荐）

```python
from b1k.vision import GraspPoseAdjuster
from b1k.models.r1pro.grasp_poses import grasp_poses

adjuster = GraspPoseAdjuster(
    client=client,
    robot_id=robot_id,
    object_mesh_path="b1k/models/objects/fridge_dszchb.obj",
    camera_link_name="zed_link",
)

detected_pose, fitness = adjuster.detect_object_pose(
    camera_pos=[-1.5, 0.5, 1.0],
    camera_target=[0.0, 0.0, 0.8],
    object_id=fridge_id,
)

if fitness > 0.5:
    adjusted_pose = adjuster.adjust_grasp_pose(
        base_grasp_pose=grasp_poses["open_fridge_right"],
        detected_object_pose=detected_pose,
    )
else:
    adjusted_pose = grasp_poses["open_fridge_right"]
```

### 3.3 集成示例（`load_r1pro.py`）

```python
if args.auto_detect and args.grasp_pose:
    adjuster = GraspPoseAdjuster(
        client=client,
        robot_id=robot_id,
        object_mesh_path=mesh_path,
        camera_link_name=args.camera_link or "zed_link",
    )

    detected_pose, fitness = adjuster.detect_object_pose(...)
    final_pose = adjuster.adjust_grasp_pose(grasp_poses[args.grasp_pose], detected_pose)
else:
    final_pose = grasp_poses[args.grasp_pose]
```

---

## 4. 参数与调试

### 4.1 ICP 关键参数

| 参数 | 默认 | 建议 |
|------|------|------|
| `max_correspondence_distance` | 0.05 m | 目标较小时调小到 0.02–0.03 |
| `max_iterations` | 50 | 追求高精度时增大 |
| `use_point_to_plane` | False | 有可靠法线时可开启提高精度 |

### 4.2 点云处理

```python
PointCloudProcessor(
    voxel_size=0.01,
    remove_outliers=True,
)
```

### 4.3 常见问题

| 场景 | 处理建议 |
|------|----------|
| `fitness` ≈ 0 | 调整相机视角、检查掩码、启用粗对齐、调小 `max_correspondence_distance` |
| 相机 link 不存在 | 使用 `demo_icp.py` 自动提示候选 link，或查 URDF |
| 点云稀疏/噪声大 | 调整距离、分辨率、`voxel_size`，并开启去噪 |
| 需要可视化 | 使用 Open3D：`o3d.visualization.draw_geometries([...])` |

### 4.4 调试技巧

```python
cv2.imwrite("rgb.png", rgb)
cv2.imwrite("depth.png", (depth * 255).astype(np.uint8))
cv2.imwrite("mask.png", mask.astype(np.uint8) * 255)
print(f"Fitness: {fitness:.3f}")
print(f"Detected position: {pose[:3, 3]}")
```

---

## 5. 视觉管线详解

### 5.1 流程

```
RGBD 捕获 → 物体分割 → 点云构建 → 粗对齐 + ICP → 坐标转换 → 姿态调整
```

### 5.2 关键步骤摘要

1. **RGBD 捕获**：`RGBDCamera.capture()` 返回 RGB、深度、视图矩阵和投影矩阵。
2. **物体分割**：可选择 PyBullet segmentation 或 SAM；默认使用前者。
3. **点云处理**：`extract_object_pointcloud()` 依据相机内参构建点云，`PointCloudProcessor.preprocess()` 负责降采样、去噪、法线估计。
4. **ICP**：`coarse_alignment_ransac()` 提供初始变换，`estimate_pose_icp()` 输出最终匹配矩阵与 `fitness`。
5. **坐标转换**：`transform_to_robot_base()` 根据 `camera_to_base` 变换得到基座坐标系姿态。
6. **姿态调整**：`adjust_grasp_pose()` 根据位置偏移修正机器人底盘与关节角；必要时可结合 IK 微调。

### 5.3 重要公式

- 深度反归一化：`z = far * near / (far - (far - near) * depth_buffer)`
- 像素到相机坐标：`x = (u - cx) * z / fx`，`y = (v - cy) * z / fy`
- 坐标变换：`T_base^object = T_base^camera · T_camera^object`
- 位置偏移：`Δp = p_detected - p_assumed`

---

## 6. 代码组织与维护

### 6.1 分层架构

| 层级 | 内容 |
|------|------|
| 应用层 | `demo_icp.py`、`load_r1pro.py` |
| 接口层 | `GraspPoseAdjuster` |
| 功能层 | camera / segmentation / pointcloud / icp / transforms |
| 基础层 | PyBullet、Open3D、SAM、NumPy 等 |

### 6.2 维护建议

- 新增物体：在 `object_configs.py` 注册 mesh 与默认相机视角，并在 `grasp_poses.py` 增加基础姿态。
- 扩展流程：对 `pose_adjuster.py` 进行子类化，插入自定义粗对齐或 IK 步骤。
- 测试建议：创建 `tests/test_vision/`，覆盖 RGBD 捕获、分割、ICP 和姿态调整。

---

## 7. 性能基准

| 步骤 | 时间 (MacBook Pro M1) |
|------|-----------------------|
| RGBD 捕获 | ~10 ms |
| PyBullet 分割 | ~5 ms |
| SAM 分割 | ~1–2 s |
| 点云处理 | ~50 ms |
| ICP 匹配 | ~100–200 ms |
| **总计** | **~200 ms**（使用 PyBullet 分割） |

### 精度指标

- 位置误差约 1–2 cm
- 姿态误差约 2–5°
- 建议 `fitness ≥ 0.7`

---

## 8. 未来扩展计划

- [ ] 调优 ICP 参数，提升鲁棒性
- [ ] 丰富物体与姿态库
- [ ] 深度集成 `load_r1pro.py`
- [ ] 补充单元测试与 CI
- [ ] 探索多相机融合、真实机器人实验

---

## 9. 常用命令速查

```bash
python scripts/demo_icp.py --object fridge
python scripts/demo_icp.py --object fridge --no-auto-detect
python scripts/demo_icp.py --object fridge --camera-link right_realsense_link
python scripts/demo_icp.py --object frying_pan --use-sam --sam-checkpoint sam.pth
```

---

## 10. 参考资料

- `docs/vision_based_grasp_adjustment.md`：动态抓取姿态策略、公式与流程详解。
- `docs/code_organization.md`：模块拆分、重构原则与维护注意事项。
- `docs/icp_usage_guide.md`：API 文档、FAQ、调试技巧。
- `docs/icp_quick_reference.md`：命令与参数速查表。

> 本文件即为上述所有文档的归纳整合版，可替代逐篇查阅。

