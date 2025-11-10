# R1Pro 机器人 PyBullet 加载指南

本目录包含使用 PyBullet 加载和显示 r1pro 机器人的脚本和资源。

## 文件说明

- `r1pro_source.urdf` - 机器人 URDF 模型文件
- `r1pro_source_cfg.yaml` - 机器人配置文件（用于其他框架）
- `constants.py` - 机器人关节常量定义
- `color_urdf.py` - URDF 着色工具脚本
- `load_r1pro.py` - 基础加载脚本
- `load_r1pro_advanced.py` - 高级加载脚本（带关节控制）

## 快速开始

### 方法 1: 基础加载脚本

最简单的使用方式：

```bash
python load_r1pro.py
```

这将：
- 连接到 PyBullet GUI
- 加载地面
- 加载 r1pro 机器人
- 应用初始关节状态（如果存在）
- 开始物理仿真

### 方法 2: 高级加载脚本（推荐）

包含交互式关节控制：

```bash
python load_r1pro_advanced.py
```

功能：
- 所有基础功能
- 显示所有关节信息
- 交互式滑块控制每个关节
- 实时位置反馈

## 代码示例

### 基本使用

```python
import pybullet as p
import pybullet_data

# 连接到 GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置重力
p.setGravity(0, 0, -9.81)

# 加载地面
p.loadURDF("plane.urdf")

# 加载机器人
robot_id = p.loadURDF("r1pro_source.urdf", [0, 0, 0.5])

# 运行仿真
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)
```

### 控制关节

```python
from load_r1pro_advanced import R1ProRobot

# 创建机器人实例
robot = R1ProRobot("r1pro_source.urdf")
robot.load()

# 设置关节位置（使用弧度）
robot.set_joint_positions({
    "torso_joint1": -0.524,  # -30度
    "torso_joint2": 1.571,   # 90度
})

# 控制关节运动
robot.set_joint_control("left_arm_joint1", 0.5, p.POSITION_CONTROL)
```

### 获取关节信息

```python
# 获取所有关节位置
positions = robot.get_joint_positions()

# 获取特定关节位置
torso_pos = robot.get_joint_positions(["torso_joint1", "torso_joint2"])

# 打印机器人信息
robot.print_info()
```

## 控制说明

### 鼠标操作
- **左键拖动**: 旋转视角
- **右键拖动**: 平移视角
- **滚轮**: 缩放

### 键盘操作
- **空格键**: 暂停/继续仿真
- **ESC 键**: 退出

### 滑块控制（高级版本）
右侧面板显示所有可控制关节的滑块，可以直接拖动调整关节位置。

## 关节说明

机器人包含以下主要关节组：

### 躯干关节 (Torso)
- `torso_joint1` - 第一个躯干关节
- `torso_joint2` - 第二个躯干关节
- `torso_joint3` - 第三个躯干关节
- `torso_joint4` - 第四个躯干关节

### 左臂关节 (Left Arm)
- `left_arm_joint1` 到 `left_arm_joint7` - 7个左臂关节
- `left_gripper_finger_joint1`, `left_gripper_finger_joint2` - 左夹爪关节

### 右臂关节 (Right Arm)
- `right_arm_joint1` 到 `right_arm_joint7` - 7个右臂关节
- `right_gripper_finger_joint1`, `right_gripper_finger_joint2` - 右夹爪关节

### 底盘关节 (Base)
- `steer_motor_joint1`, `steer_motor_joint2`, `steer_motor_joint3` - 转向关节
- `wheel_motor_joint1`, `wheel_motor_joint2`, `wheel_motor_joint3` - 轮子关节

## 常见问题

### 1. URDF 文件找不到

确保 `r1pro_source.urdf` 和 `meshes/` 目录在同一目录下。

### 2. Mesh 文件找不到

检查 `meshes/` 目录是否存在，并且 URDF 中的 mesh 路径是否正确。

### 3. 机器人掉到地面以下

调整初始位置：
```python
robot = R1ProRobot(urdf_path, start_pos=[0, 0, 0.5])
```

### 4. 关节控制不响应

检查关节是否在可控制列表中：
```python
robot.print_info()  # 查看所有可控制关节
```

## 预设关节状态

`constants.py` 中定义了两个预设状态：

- `r1pro_init_joint_state` - 初始状态（双臂下垂）
- `r1pro_T_joint_state` - T 形状态（双臂展开）

使用示例：
```python
from constants import r1pro_init_joint_state
robot.set_joint_positions(r1pro_init_joint_state)
```

## 自定义配置

### 固定基座

如果想让机器人基座固定不动：

```python
robotId = p.loadURDF(
    urdf_path,
    useFixedBase=True  # 固定基座
)
```

### 禁用自碰撞

```python
robotId = p.loadURDF(
    urdf_path,
    flags=0  # 不使用自碰撞标志
)
```

### 自定义摄像头位置

```python
p.resetDebugVisualizationCamera(
    cameraDistance=5.0,      # 距离
    cameraYaw=90,             # 偏航角
    cameraPitch=-45,          # 俯仰角
    cameraTargetPosition=[1, 1, 1]  # 目标位置
)
```

## 下一步

- 查看 `examples/pybullet/examples/` 目录了解更多 PyBullet 示例
- 阅读 [PyBullet 文档](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit)
- 探索机器人运动学和控制算法

## 机器人稳定落地配置

`load_r1pro_advanced.py` 脚本已自动配置了多项稳定化措施，确保机器人平稳落地不摔倒：

### 已实现的稳定化功能

1. **物理参数配置**
   - 侧向摩擦系数：1.0（防止滑动）
   - 线性/角阻尼：0.04（减少震荡）
   - 关节阻尼：0.1（减少摆动）

2. **降低初始高度**
   - 从 0.5 米降低到 0.15 米
   - 减少落地冲击

3. **关节稳定控制**
   - 使用位置控制保持关节姿态
   - 防止因重力导致的摆动

4. **预稳定阶段**
   - 自动运行 100 步让机器人稳定落地
   - 确保完全稳定后再开始交互

### 自定义稳定参数

如果需要调整稳定参数：

```python
# 在加载机器人后
robot.configure_stability(
    lateral_friction=1.5,      # 增加摩擦（更稳定）
    linear_damping=0.06,        # 增加阻尼（更稳定但反应慢）
    angular_damping=0.06,
    joint_damping=0.15
)

# 调整关节稳定刚度
robot.stabilize_joints(stiffness=80.0)  # 更高的刚度 = 更强的保持力
```

### 故障排除

**机器人仍然摔倒**:
- 增加摩擦系数: `lateral_friction=1.5`
- 增加阻尼: `linear_damping=0.08, angular_damping=0.08`
- 增加预稳定时间: 将循环步数从 100 增加到 200

**机器人太僵硬**:
- 减少阻尼参数
- 降低关节刚度: `stiffness=20.0`

**固定基座**（完全防止摔倒，但无法移动）:
```python
robotId = p.loadURDF(urdf_path, ..., useFixedBase=True)
```

更多详细说明请参考代码注释。

