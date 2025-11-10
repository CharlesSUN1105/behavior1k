#!/usr/bin/env python3
"""
PyBullet 加载和显示 r1pro 机器人（高级版本）

功能:
    - 加载机器人并显示在 GUI 中
    - 显示关节信息
    - 交互式关节控制（通过滑块）
    - 实时物理仿真

使用方法:
    python load_r1pro_advanced.py
"""

import pybullet as p
import pybullet_data
import time
import os
import sys
import math

# 添加当前目录到路径，以便导入 constants
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from constants import r1pro_init_joint_state, r1pro_T_joint_state
    from math import radians
    
    # 定义平衡姿态（躯干关节设为0，使机器人保持直立平衡）
    r1pro_balanced_joint_state_deg = {
        "torso_joint1": 0.0,      # 躯干关节设为0，保持直立
        "torso_joint2": 0.0,
        "torso_joint3": 0.0,
        "torso_joint4": 0.0,
        "left_arm_joint1": 0.0,
        "left_arm_joint2": 0.0,
        "left_arm_joint3": 0.0,
        "left_arm_joint4": -90.0,  # 手臂稍微下垂，保持平衡
        "left_arm_joint5": 0.0,
        "left_arm_joint6": 0.0,
        "left_arm_joint7": 0.0,
        "right_arm_joint1": 0.0,
        "right_arm_joint2": 0.0,
        "right_arm_joint3": 0.0,
        "right_arm_joint4": -90.0,  # 手臂稍微下垂，保持平衡
        "right_arm_joint5": 0.0,
        "right_arm_joint6": 0.0,
        "right_arm_joint7": 0.0,
    }
    
    r1pro_balanced_joint_state = {}
    for k, v in r1pro_balanced_joint_state_deg.items():
        r1pro_balanced_joint_state[k] = radians(v)
    
    # 定义更自然的“放松”站姿，让躯干略微前倾、手臂轻微弯曲
    r1pro_relaxed_joint_state_deg = dict(r1pro_balanced_joint_state_deg)
    r1pro_relaxed_joint_state_deg.update({
        "torso_joint1": 4.0,
        "torso_joint2": -6.0,
        "torso_joint3": 3.5,
        "torso_joint4": 8.0,
        "left_arm_joint1": 12.0,
        "left_arm_joint2": -18.0,
        "left_arm_joint3": 10.0,
        "left_arm_joint4": -70.0,
        "left_arm_joint5": -12.0,
        "left_arm_joint6": 8.0,
        "left_arm_joint7": -5.0,
        "right_arm_joint1": -12.0,
        "right_arm_joint2": 18.0,
        "right_arm_joint3": -10.0,
        "right_arm_joint4": -70.0,
        "right_arm_joint5": 12.0,
        "right_arm_joint6": -8.0,
        "right_arm_joint7": 5.0,
    })
    
    r1pro_relaxed_joint_state = {}
    for k, v in r1pro_relaxed_joint_state_deg.items():
        r1pro_relaxed_joint_state[k] = radians(v)
    
    # 定义低位抓取姿态：躯干明显下弯，双臂前探降低末端位置
    r1pro_low_reach_joint_state_deg = dict(r1pro_balanced_joint_state_deg)
    r1pro_low_reach_joint_state_deg.update({
        "torso_joint1": 24.0,     # 让底部杆件向前弯折
        "torso_joint2": -60.0,    # 中段向后弯折以形成“S”形支撑
        "torso_joint3": -36.0,    # 反向补偿，保持上躯干基本竖直
        "torso_joint4": 0.0,
        "left_arm_joint1": 6.0,
        "left_arm_joint2": -24.0,
        "left_arm_joint3": 8.0,
        "left_arm_joint4": -60.0,
        "left_arm_joint5": -14.0,
        "left_arm_joint6": 8.0,
        "left_arm_joint7": -5.0,
        "right_arm_joint1": -6.0,
        "right_arm_joint2": 24.0,
        "right_arm_joint3": -8.0,
        "right_arm_joint4": -60.0,
        "right_arm_joint5": 14.0,
        "right_arm_joint6": -8.0,
        "right_arm_joint7": 5.0,
    })
    
    r1pro_low_reach_joint_state = {}
    for k, v in r1pro_low_reach_joint_state_deg.items():
        r1pro_low_reach_joint_state[k] = radians(v)
    
    r1pro_pose_profiles = {
        "balanced": r1pro_balanced_joint_state,
        "relaxed": r1pro_relaxed_joint_state,
        "low_reach": r1pro_low_reach_joint_state,
        "init": r1pro_init_joint_state,
        "t_pose": r1pro_T_joint_state,
    }
except ImportError:
    print("警告: 无法导入 constants.py，将使用默认关节状态")
    r1pro_init_joint_state = {}
    r1pro_T_joint_state = {}
    r1pro_balanced_joint_state = {}
    r1pro_relaxed_joint_state = {}
    r1pro_low_reach_joint_state = {}
    r1pro_pose_profiles = {}


class R1ProRobot:
    """R1Pro 机器人控制类"""
    
    def __init__(self, urdf_path, start_pos=None, start_orientation=None, fixed_base=False):
        """
        初始化机器人
        
        Args:
            urdf_path: URDF 文件路径
            start_pos: 初始位置 [x, y, z]
            start_orientation: 初始朝向（四元数），如果为 None 则使用 [0,0,0]
            fixed_base: 是否以固定基座加载机器人
        """
        self.urdf_path = urdf_path
        self.start_pos = list(start_pos) if start_pos is not None else [0, 0, 0.5]
        self.start_orientation = start_orientation or p.getQuaternionFromEuler([0, 0, 0])
        self.fixed_base = fixed_base
        self.robotId = None
        self.joint_name_to_id = {}
        self.joint_id_to_name = {}
        self.controllable_joints = []
        self.initial_joint_positions = {}  # 保存初始关节位置
        self.locked_joints = set()  # 存储需要锁定的关节名称集合
        self.active_control_joints = set()  # 当前由 position/velocity 控制的关节
        self.base_constraint_id = None  # 世界坐标下固定基座的约束ID
        self.original_base_mass = None
        self.original_base_inertia = None
        self.default_position_gain = 0.6
        self.default_velocity_gain = 1.0
        self.max_control_velocity = 1.5
        
    def load(self):
        """加载机器人到仿真环境"""
        print(f"正在加载 URDF: {self.urdf_path}")
        
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF 文件不存在: {self.urdf_path}")
        
        self.robotId = p.loadURDF(
            self.urdf_path,
            self.start_pos,
            self.start_orientation,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase=self.fixed_base
        )
        
        if self.robotId < 0:
            raise RuntimeError("无法加载机器人 URDF")
        
        # 记录基座的原始动力学属性，方便后续恢复
        base_dynamics = p.getDynamicsInfo(self.robotId, -1)
        if base_dynamics:
            self.original_base_mass = base_dynamics[0]
            self.original_base_inertia = list(base_dynamics[2]) if base_dynamics[2] is not None else None
        
        self._build_joint_mapping()
        print(f"✓ 成功加载机器人，ID: {self.robotId}")
        
        # 禁用所有关节的默认电机，防止意外运动
        self.disable_default_joint_motors()
        
        # 配置物理参数以增加稳定性
        self.configure_stability()
        
        # 如果URDF没有颜色，通过代码设置颜色
        self.apply_colors_if_needed()
        
        return self.robotId
    
    def lock_joints(self, joint_names, stiffness=100.0):
        """
        锁定指定的关节，使其保持在当前位置
        
        Args:
            joint_names: 关节名称列表或集合，要锁定的关节
            stiffness: 位置控制的刚度（力大小，百分比）
        """
        if isinstance(joint_names, str):
            joint_names = [joint_names]
        
        for joint_name in joint_names:
            if joint_name in self.joint_name_to_id:
                self.locked_joints.add(joint_name)
                # 立即锁定当前位置
                current_pos = self.get_joint_positions([joint_name])[joint_name]
                self.set_joint_control(joint_name, current_pos, p.POSITION_CONTROL, stiffness=stiffness)
                print(f"  ✓ 已锁定关节: {joint_name} (位置: {current_pos:.4f} rad)")
            else:
                print(f"  ⚠ 警告: 未知关节名称: {joint_name}")
    
    def unlock_joints(self, joint_names=None):
        """
        解锁指定的关节
        
        Args:
            joint_names: 关节名称列表或集合，要解锁的关节。如果为 None，则解锁所有关节
        """
        if joint_names is None:
            # 解锁所有关节
            self.locked_joints.clear()
            print("  ✓ 已解锁所有关节")
        else:
            if isinstance(joint_names, str):
                joint_names = [joint_names]
            
            for joint_name in joint_names:
                if joint_name in self.locked_joints:
                    self.locked_joints.remove(joint_name)
                    self.active_control_joints.discard(joint_name)
                    print(f"  ✓ 已解锁关节: {joint_name}")
                else:
                    print(f"  ⚠ 关节 {joint_name} 未锁定")
    
    def get_locked_joints(self):
        """返回当前锁定的关节列表"""
        return list(self.locked_joints)
    
    def hold_locked_joints(self, stiffness=100.0):
        """
        保持所有锁定的关节在当前位置
        
        Args:
            stiffness: 位置控制的刚度（力大小，百分比）
        """
        for joint_name in self.locked_joints:
            if joint_name in self.joint_name_to_id:
                # 获取当前位置
                current_pos = self.get_joint_positions([joint_name])[joint_name]
                # 应用位置控制保持当前位置
                self.set_joint_control(joint_name, current_pos, p.POSITION_CONTROL, stiffness=stiffness)
    
    def lock_joints_by_pattern(self, pattern, stiffness=100.0):
        """
        根据模式锁定关节（例如：锁定所有包含 "torso" 的关节）
        
        Args:
            pattern: 字符串模式，匹配的关节名称将被锁定
            stiffness: 位置控制的刚度（力大小，百分比）
        """
        matched_joints = []
        for joint_name in self.joint_name_to_id.keys():
            if pattern.lower() in joint_name.lower():
                matched_joints.append(joint_name)
        
        if matched_joints:
            self.lock_joints(matched_joints, stiffness=stiffness)
            print(f"  ✓ 根据模式 '{pattern}' 锁定了 {len(matched_joints)} 个关节")
        else:
            print(f"  ⚠ 未找到匹配模式 '{pattern}' 的关节")
        
        return matched_joints
    
    def lock_all_except_arms_and_grippers(self, stiffness=100.0):
        """
        锁定除手臂和夹爪之外的所有关节
        
        Args:
            stiffness: 位置控制的刚度（力大小，百分比）
        """
        print("正在锁定除手臂和夹爪之外的所有关节...")
        
        # 获取所有可控制关节
        all_joints = set(self.joint_name_to_id.keys())
        
        # 定义要保留的关节（手臂和夹爪）
        # 手臂关节：包含 "arm" 但不包含 "gripper"（gripper关节单独处理）
        # 夹爪关节：包含 "gripper" 或 "finger"
        joints_to_keep = set()
        
        for joint_name in all_joints:
            joint_lower = joint_name.lower()
            # 保留手臂关节（left_arm_joint*, right_arm_joint*）
            if ("arm" in joint_lower and "gripper" not in joint_lower) or \
               ("gripper" in joint_lower) or \
               ("finger" in joint_lower):
                joints_to_keep.add(joint_name)
        
        # 锁定所有不在保留列表中的关节
        joints_to_lock = all_joints - joints_to_keep
        
        if joints_to_lock:
            self.lock_joints(list(joints_to_lock), stiffness=stiffness)
            print(f"  ✓ 已锁定 {len(joints_to_lock)} 个关节")
            print(f"  ✓ 保留 {len(joints_to_keep)} 个关节可控（手臂和夹爪）")
        else:
            print("  ⚠ 未找到需要锁定的关节")
        
        return list(joints_to_lock), list(joints_to_keep)
    
    def get_pose_profile(self, profile_name):
        """
        获取预定义姿态配置
        
        Args:
            profile_name: 姿态名称（balanced/relaxed/init/t_pose等）
        
        Returns:
            字典形式的关节目标角度（弧度），若不存在返回 None
        """
        if not profile_name:
            return None
        
        profile_key = profile_name.lower()
        if profile_key in r1pro_pose_profiles and r1pro_pose_profiles[profile_key]:
            return r1pro_pose_profiles[profile_key]
        
        print(f"  ⚠ 未找到姿态配置 '{profile_name}'，使用当前关节状态")
        return None
    
    def fix_base_to_world(self):
        """通过约束将基座固定在当前世界位姿"""
        if self.fixed_base:
            return None
        if self.base_constraint_id is not None:
            return self.base_constraint_id
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId)
        self.base_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robotId,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=base_pos,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=base_orn
        )
        print("  ✓ 已通过世界约束固定基座")
        return self.base_constraint_id
    
    def release_base_constraint(self):
        """释放固定基座的世界约束"""
        if self.base_constraint_id is not None:
            p.removeConstraint(self.base_constraint_id)
            self.base_constraint_id = None
            print("  ✓ 已释放基座约束")
    
    def initialize_for_grasp_annotations(
        self,
        ground_height=0.0,
        pose_profile="balanced",
        lock_non_arm_joints=True,
        stiffness=120.0,
        settle_steps=120,
        use_fixed_base_during_setup=None,
        lock_base=True
    ):
        """
        将机器人快速设置到抓取标注使用的稳定姿态。
        
        Args:
            ground_height: 目标地面高度。
            pose_profile: 姿态配置名称（balanced、relaxed、init、t_pose 等）。
            lock_non_arm_joints: 是否锁定除手臂/夹爪以外的关节。
            stiffness: 位置控制的刚度百分比。
            settle_steps: 额外的仿真步数，用于等待姿态稳定。
            use_fixed_base_during_setup: 放置到地面时是否临时固定基座。
            lock_base: 对非固定基座机器人，是否添加世界约束锁定基座。
        
        Returns:
            dict: 包含 locked/free 关节名称列表。
        """
        if use_fixed_base_during_setup is None:
            use_fixed_base_during_setup = not self.fixed_base
        
        print("\n正在准备抓取标注所需的稳定姿态...")
        
        pose_profile = (pose_profile or "balanced").lower()
        pose_joint_positions = self.get_pose_profile(pose_profile)
        
        if pose_joint_positions:
            self.set_joint_positions(pose_joint_positions)
            print(f"  ✓ 使用 '{pose_profile}' 姿态配置")
        elif r1pro_init_joint_state:
            self.set_joint_positions(r1pro_init_joint_state)
            print("  ✓ 使用初始姿态作为起点")
        else:
            self.save_current_joint_positions()
            print("  ⚠ 未找到预设姿态，使用当前关节状态")
        
        self.disable_default_joint_motors()
        
        self.place_on_ground(
            ground_height=ground_height,
            use_fixed_base_during_setup=use_fixed_base_during_setup
        )
        p.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0])
        
        locked_joints = []
        free_joints = list(self.joint_name_to_id.keys())
        
        if lock_non_arm_joints:
            locked_joints, free_joints = self.lock_all_except_arms_and_grippers(stiffness=stiffness)
        
        self.save_current_joint_positions()
        self.disable_default_joint_motors()
        self.hold_initial_positions(stiffness=stiffness)
        self.hold_locked_joints(stiffness=stiffness)
        
        if lock_base and not self.fixed_base:
            self.fix_base_to_world()
        
        steps = max(0, int(settle_steps))
        for _ in range(steps):
            self.disable_default_joint_motors()
            self.hold_initial_positions(stiffness=stiffness)
            self.hold_locked_joints(stiffness=stiffness)
            p.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()
        
        p.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0])
        print("  ✓ 机器人已稳定就绪，可开始标注抓取姿态")
        
        return {"locked": locked_joints, "free": free_joints}
    
    def disable_default_joint_motors(self, skip_names=None):
        """禁用所有关节的默认电机，防止因重力导致的意外运动"""
        
        disabled_count = 0
        num_joints = p.getNumJoints(self.robotId)
        skip_set = set(skip_names or [])
        skip_set.update(self.locked_joints)
        skip_set.update(self.active_control_joints)
        
        # 禁用所有关节的默认电机（包括可控制和不可控制的）
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8') if joint_info[1] else ""
            
            if joint_name in skip_set:
                continue
            
            # 对所有旋转和滑动关节禁用默认电机
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                # 设置为速度控制模式，但目标速度为0，力为0（相当于禁用）
                p.setJointMotorControl2(
                    self.robotId,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )
                disabled_count += 1
        
        
    
    def place_on_ground(self, ground_height=0.0, use_fixed_base_during_setup=False):
        """
        将机器人放置在 ground_height 高度的地面上，使底盘刚好接触地面
        
        Args:
            ground_height: 地面高度（默认 0.0）
            use_fixed_base_during_setup: 是否在设置期间临时固定基座（默认 False）
        """
        print("正在计算机器人底盘高度...")
        
        # 如果需要在设置期间固定基座
        backup_dynamics = None
        if use_fixed_base_during_setup:
            # 记录原始动力学参数并临时固定基座
            base_dynamics = p.getDynamicsInfo(self.robotId, -1)
            backup_mass = base_dynamics[0] if base_dynamics else self.original_base_mass
            backup_inertia = list(base_dynamics[2]) if base_dynamics and base_dynamics[2] is not None else self.original_base_inertia
            backup_dynamics = (backup_mass, backup_inertia)
            p.changeDynamics(self.robotId, -1, mass=0, localInertiaDiagonal=[0, 0, 0])
            print("  ⚠ 临时固定基座以确保稳定")
        
        # 获取基座（base link）的AABB（轴对齐包围盒）来找到最低点
        # 使用 -1 表示基座链接，而不是整个机器人
        aabb_min, aabb_max = p.getAABB(self.robotId, -1)
        base_bottom = aabb_min[2]  # Z轴最小值是底盘底部
        
        # 获取当前基座位置（基座中心的位置）
        current_pos, current_orn = p.getBasePositionAndOrientation(self.robotId)
        
        # 计算需要调整的高度，使底盘底部刚好接触地面
        # offset_z = 底盘底部相对于基座中心的高度（通常是负值，因为底部在中心下方）
        offset_z = base_bottom - current_pos[2]
        
        # 目标位置 = 地面高度 - offset_z
        # 例如：如果 offset_z = -0.1719，ground_height = 0.0
        # 则 target_z = 0.0 - (-0.1719) = 0.1719
        # 这样基座中心在 0.1719 米，底盘底部在 0.1719 + (-0.1719) = 0.0 米
        target_z = ground_height - offset_z
        
        # 打印调试信息
        print(f"  基座中心当前位置: {current_pos[2]:.4f} 米")
        print(f"  底盘底部位置 (AABB min): {base_bottom:.4f} 米")
        print(f"  底盘底部相对基座中心的偏移: {offset_z:.4f} 米")
        print(f"  计算得到的目标基座中心高度: {target_z:.4f} 米")
        print(f"  这将使底盘底部在: {target_z + offset_z:.4f} 米（应该等于地面高度 {ground_height:.4f} 米）")
        
        # 移除之前的安全检查，直接使用计算结果
        # 之前的代码会强制抬高0.01米，这是不必要的
        
        # 设置新的基座位置
        new_pos = [current_pos[0], current_pos[1], target_z]
        p.resetBasePositionAndOrientation(self.robotId, new_pos, current_orn)
        
        # 重置速度，确保静止
        p.resetBaseVelocity(self.robotId, [0, 0, 0], [0, 0, 0])
        
        # 如果临时固定了基座，运行几步后再恢复
        if use_fixed_base_during_setup:
            for _ in range(10):
                p.stepSimulation()
                # 持续应用位置控制
                self.hold_initial_positions(stiffness=100.0)
            
            if backup_dynamics and not self.fixed_base:
                restore_mass, restore_inertia = backup_dynamics
                kwargs = {"mass": restore_mass if restore_mass is not None else 0}
                if restore_inertia is not None:
                    kwargs["localInertiaDiagonal"] = restore_inertia
                p.changeDynamics(self.robotId, -1, **kwargs)
                print("  ✓ 已恢复基座原始动力学参数")
        
        # 验证最终位置
        final_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        final_aabb_min, _ = p.getAABB(self.robotId, -1)
        final_bottom = final_aabb_min[2]
        print(f"  ✓ 机器人已放置在高度 {target_z:.4f} 米")
        print(f"  ✓ 底盘底部实际位置: {final_bottom:.4f} 米（目标: {ground_height:.4f} 米）")
        
        return new_pos
    
    def apply_colors_if_needed(self):
        """如果URDF没有颜色，通过代码为机器人设置颜色"""
        print("正在检查并应用颜色...")
        
        # 检查基座是否有颜色（通过获取visual shape信息）
        try:
            visual_shape_data = p.getVisualShapeData(self.robotId, -1)
            # getVisualShapeData返回格式: (objectUniqueId, linkIndex, visualGeometryType, dimensions, localFramePos, localFrameOrn, rgbaColor, textureUniqueId)
            # rgba颜色在索引6（不是7）
            if visual_shape_data and len(visual_shape_data) > 0:
                rgba = visual_shape_data[0][6]  # rgba颜色在索引6
                # 如果颜色接近白色（默认值），说明没有设置颜色
                if rgba[0] > 0.95 and rgba[1] > 0.95 and rgba[2] > 0.95:
                    print("  URDF文件没有颜色定义，正在应用默认颜色...")
                    self.apply_default_colors()
                else:
                    print(f"  ✓ URDF文件已包含颜色定义（基座颜色: RGB({rgba[0]:.2f}, {rgba[1]:.2f}, {rgba[2]:.2f})）")
        except Exception as e:
            # 如果检查失败，默认应用颜色
            print(f"  无法检查颜色，应用默认颜色... (错误: {e})")
            self.apply_default_colors()
    
    def apply_default_colors(self):
        """为机器人各部分应用默认颜色方案"""
        # 定义颜色方案（RGBA格式）
        color_scheme = {
            'base_link': [0.7, 0.7, 0.7, 1.0],  # 灰色 - 基座
            'torso': [0.4, 0.4, 0.4, 1.0],  # 深灰色 - 躯干
            'left_arm': [0.3, 0.7, 0.3, 1.0],  # 绿色 - 左臂
            'right_arm': [0.3, 0.3, 0.8, 1.0],  # 蓝色 - 右臂
            'left_gripper': [0.8, 0.3, 0.3, 1.0],  # 红色 - 左夹爪
            'right_gripper': [0.8, 0.3, 0.3, 1.0],  # 红色 - 右夹爪
            'wheel': [0.6, 0.6, 0.7, 1.0],  # 金属色 - 轮子
            'default': [0.9, 0.9, 0.9, 1.0]  # 浅灰色 - 默认
        }
        
        num_joints = p.getNumJoints(self.robotId)
        
        # 为基座设置颜色
        p.changeVisualShape(self.robotId, -1, rgbaColor=color_scheme['base_link'])
        
        # 为各个链接设置颜色
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            link_name = joint_info[12].decode('utf-8') if joint_info[12] else ""
            
            # 根据链接名称选择颜色
            if 'torso' in link_name.lower():
                color = color_scheme['torso']
            elif 'left_arm' in link_name.lower():
                color = color_scheme['left_arm']
            elif 'left_gripper' in link_name.lower():
                color = color_scheme['left_gripper']
            elif 'right_arm' in link_name.lower():
                color = color_scheme['right_arm']
            elif 'right_gripper' in link_name.lower():
                color = color_scheme['right_gripper']
            elif 'wheel' in link_name.lower() or 'steer' in link_name.lower():
                color = color_scheme['wheel']
            else:
                color = color_scheme['default']
            
            # 应用颜色
            p.changeVisualShape(self.robotId, i, rgbaColor=color)
        
        print(f"  ✓ 已为 {num_joints + 1} 个链接应用颜色")
    
    def configure_stability(self, 
                            lateral_friction=1.5,
                            linear_damping=0.08,
                            angular_damping=0.08,
                            joint_damping=0.15):
        """
        配置机器人物理参数以增加稳定性
        
        Args:
            lateral_friction: 侧向摩擦系数（默认 1.0，增加可以防止滑动）
            linear_damping: 线性阻尼（默认 0.04，减少平移运动）
            angular_damping: 角阻尼（默认 0.04，减少旋转运动）
            joint_damping: 关节阻尼（默认 0.1，减少关节摆动）
        """
        print("正在配置机器人物理参数以提高稳定性...")
        
        # 设置基座（base link）的摩擦和阻尼
        p.changeDynamics(
            self.robotId,
            -1,  # -1 表示基座
            lateralFriction=lateral_friction,
            linearDamping=linear_damping,
            angularDamping=angular_damping
        )
        
        # 设置所有链接的摩擦和阻尼
        num_joints = p.getNumJoints(self.robotId)
        for i in range(num_joints):
            p.changeDynamics(
                self.robotId,
                i,
                lateralFriction=lateral_friction,
                linearDamping=linear_damping,
                angularDamping=angular_damping
            )
        
        # 设置所有关节的阻尼（注意：关节摩擦需要通过 URDF 或其他方式设置）
        for joint in self.controllable_joints:
            joint_id = joint['id']
            # 注意：jointDamping 需要指定 linkIndex，应该是关节连接的子链接
            # 但通常我们使用 joint_id 作为 linkIndex
            p.changeDynamics(
                self.robotId,
                joint_id,  # linkIndex - 关节连接的子链接
                jointDamping=joint_damping
            )
        
        print(f"  ✓ 摩擦系数: {lateral_friction}")
        print(f"  ✓ 线性阻尼: {linear_damping}")
        print(f"  ✓ 角阻尼: {angular_damping}")
        print(f"  ✓ 关节阻尼: {joint_damping}")
    
    def stabilize_joints(self, stiffness=100.0):
        """
        使用位置控制稳定关节，防止因重力导致的摆动
        
        Args:
            stiffness: 位置控制的刚度（力大小，百分比）
        """
        # 只稳定可控制的关节
        for joint in self.controllable_joints:
            joint_id = joint['id']
            joint_name = joint['name']
            
            # 获取当前关节位置
            joint_state = p.getJointState(self.robotId, joint_id)
            current_position = joint_state[0]
            
            # 使用位置控制保持当前位置
            max_force = joint['max_force']
            force = max_force * (stiffness / 100.0)
            
            p.setJointMotorControl2(
                self.robotId,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=current_position,
                force=force
            )
    
    def _build_joint_mapping(self):
        """构建关节名称到 ID 的映射"""
        num_joints = p.getNumJoints(self.robotId)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            self.joint_name_to_id[joint_name] = i
            self.joint_id_to_name[i] = joint_name
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joints.append({
                    'id': i,
                    'name': joint_name,
                    'type': joint_type,
                    'lower_limit': joint_info[8],
                    'upper_limit': joint_info[9],
                    'max_force': joint_info[10] if joint_info[10] > 0 else 100
                })
        
        print(f"✓ 找到 {len(self.controllable_joints)} 个可控制关节")
    
    def set_joint_positions(self, joint_positions):
        """
        设置关节位置
        
        Args:
            joint_positions: 字典，键为关节名称，值为位置（弧度）
        """
        for joint_name, position in joint_positions.items():
            if joint_name in self.joint_name_to_id:
                joint_id = self.joint_name_to_id[joint_name]
                p.resetJointState(self.robotId, joint_id, position)
                # 保存初始位置
                self.initial_joint_positions[joint_name] = position
            else:
                print(f"警告: 未知关节名称: {joint_name}")
    
    def save_current_joint_positions(self):
        """保存当前所有关节位置作为初始位置"""
        self.initial_joint_positions = self.get_joint_positions()
    
    def hold_initial_positions(self, stiffness=100.0):
        """
        保持初始关节位置，防止因重力导致的摆动
        
        Args:
            stiffness: 位置控制的刚度（力大小，百分比）
        """
        # 如果没有初始位置，先保存当前位置
        if not self.initial_joint_positions:
            self.save_current_joint_positions()
        
        # 保持初始位置
        for joint_name, target_position in self.initial_joint_positions.items():
            if joint_name in self.joint_name_to_id:
                joint_id = self.joint_name_to_id[joint_name]
                joint_info = next((j for j in self.controllable_joints if j['name'] == joint_name), None)
                if joint_info:
                    max_force = joint_info['max_force']
                    force = max_force * (stiffness / 100.0)
                    
                    p.setJointMotorControl2(
                        self.robotId,
                        joint_id,
                        p.POSITION_CONTROL,
                        targetPosition=target_position,
                        force=force
                    )
    
    def get_joint_positions(self, joint_names=None):
        """
        获取关节位置
        
        Args:
            joint_names: 关节名称列表，如果为 None 则返回所有关节
        
        Returns:
            字典，键为关节名称，值为位置（弧度）
        """
        if joint_names is None:
            joint_names = list(self.joint_name_to_id.keys())
        
        positions = {}
        for joint_name in joint_names:
            if joint_name in self.joint_name_to_id:
                joint_id = self.joint_name_to_id[joint_name]
                joint_state = p.getJointState(self.robotId, joint_id)
                positions[joint_name] = joint_state[0]
        
        return positions
    
    def set_joint_control(self, joint_name, target_position, control_mode=p.POSITION_CONTROL, stiffness=100.0):
        """
        设置关节控制
        
        Args:
            joint_name: 关节名称
            target_position: 目标位置（弧度）
            control_mode: 控制模式（POSITION_CONTROL, VELOCITY_CONTROL, TORQUE_CONTROL）
            stiffness: 位置控制的刚度百分比（仅用于 POSITION_CONTROL）
        """
        if joint_name not in self.joint_name_to_id:
            print(f"警告: 未知关节名称: {joint_name}")
            return
        
        # 检查 target_position 是否为 None
        if target_position is None:
            return
        
        joint_id = self.joint_name_to_id[joint_name]
        
        # 找到关节的最大力
        joint_info = next((j for j in self.controllable_joints if j['name'] == joint_name), None)
        max_force = joint_info['max_force'] if joint_info else 100
        self.active_control_joints.add(joint_name)
        
        # 根据控制模式设置参数
        if control_mode == p.POSITION_CONTROL:
            # 使用刚度百分比调整力大小
            force = max_force * (stiffness / 100.0)
            p.setJointMotorControl2(
                self.robotId,
                joint_id,
                control_mode,
                targetPosition=target_position,
                force=force,
                positionGain=self.default_position_gain,
                velocityGain=self.default_velocity_gain,
                maxVelocity=self.max_control_velocity
            )
        elif control_mode == p.VELOCITY_CONTROL:
            p.setJointMotorControl2(
                self.robotId,
                joint_id,
                control_mode,
                targetVelocity=target_position,
                force=max_force,
                positionGain=self.default_position_gain,
                velocityGain=self.default_velocity_gain,
                maxVelocity=self.max_control_velocity
            )
        elif control_mode == p.TORQUE_CONTROL:
            p.setJointMotorControl2(
                self.robotId,
                joint_id,
                control_mode,
                force=target_position
            )
        else:
            # 默认使用位置控制
            force = max_force * (stiffness / 100.0)
            p.setJointMotorControl2(
                self.robotId,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=force
            )
    
    def print_info(self):
        """打印机器人信息"""
        print("\n" + "=" * 60)
        print("机器人信息")
        print("=" * 60)
        print(f"机器人 ID: {self.robotId}")
        print(f"总关节数: {p.getNumJoints(self.robotId)}")
        print(f"可控制关节数: {len(self.controllable_joints)}")
        
        print("\n可控制关节列表:")
        for joint in self.controllable_joints:
            print(f"  [{joint['id']:2d}] {joint['name']:30s} "
                  f"范围: [{joint['lower_limit']:7.3f}, {joint['upper_limit']:7.3f}] "
                  f"最大力: {joint['max_force']:.1f}")


def setup_environment():
    """设置仿真环境"""
    print("正在连接到 PyBullet GUI...")
    physicsClient = p.connect(p.GUI)
    
    # 设置摄像头
    p.resetDebugVisualizerCamera(
        3.0,  # cameraDistance
        45,   # cameraYaw
        -30,  # cameraPitch
        [0, 0, 0.5]  # cameraTargetPosition
    )
    
    # 设置搜索路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置重力
    p.setGravity(0, 0, -9.81)
    
    # 加载地面
    print("正在加载地面...")
    planeId = p.loadURDF("plane.urdf")
    
    return physicsClient


def create_joint_sliders(robot):
    """创建关节控制滑块"""
    sliders = {}
    print("\n正在创建关节控制滑块...")
    
    # 为每个可控制关节创建滑块（跳过已锁定的关节）
    success_count = 0
    skipped_count = 0
    for joint in robot.controllable_joints:
        joint_name = joint['name']
        
        # 跳过已锁定的关节（不需要滑块控制）
        if joint_name in robot.locked_joints:
            skipped_count += 1
            continue
        
        lower = joint['lower_limit']
        upper = joint['upper_limit']
        
        # 检查范围是否有效（处理 lower > upper 的情况）
        if lower > upper:
            # 交换上下限
            lower, upper = upper, lower
            print(f"  注意: {joint_name} 的范围已修正为 [{lower:.3f}, {upper:.3f}]")
        
        # 获取当前关节位置，如果获取不到则使用范围中间值
        try:
            positions = robot.get_joint_positions([joint_name])
            current_pos = positions.get(joint_name, (lower + upper) / 2.0)
        except (KeyError, TypeError):
            current_pos = (lower + upper) / 2.0
        
        # 确保 current_pos 是有效的数字，并在范围内
        if current_pos is None or not isinstance(current_pos, (int, float)):
            current_pos = (lower + upper) / 2.0
        
        # 限制 current_pos 在范围内
        current_pos = max(lower, min(upper, current_pos))
        
        # 创建滑块
        try:
            slider_id = p.addUserDebugParameter(
                joint_name,
                lower,
                upper,
                current_pos
            )
            # 检查滑块是否创建成功（有效 ID 应该 >= 0）
            if slider_id is not None and slider_id >= 0:
                sliders[joint_name] = slider_id
                success_count += 1
            else:
                print(f"  警告: 滑块创建失败 ({joint_name})")
        except Exception as e:
            print(f"  警告: 无法为 {joint_name} 创建滑块: {e}")
    
    print(f"✓ 成功创建了 {success_count}/{len(robot.controllable_joints)} 个控制滑块")
    if skipped_count > 0:
        print(f"  (跳过了 {skipped_count} 个已锁定的关节)")
    return sliders


def main():
    """主函数"""
    print("=" * 60)
    print("PyBullet r1pro 机器人加载器（高级版本）")
    print("=" * 60)
    
    # 获取文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 优先使用带颜色的URDF文件，如果不存在则使用普通URDF
    urdf_path_color = os.path.join(current_dir, "r1pro_source_color.urdf")
    urdf_path = os.path.join(current_dir, "r1pro_source.urdf")
    
    # 检查是否有带颜色的URDF文件
    if os.path.exists(urdf_path_color):
        urdf_path = urdf_path_color
        print(f"使用带颜色的URDF文件: {urdf_path}")
    else:
        print(f"使用普通URDF文件: {urdf_path}")
        print("提示: 可以使用 color_urdf.py 生成带颜色的URDF文件")
    
    try:
        # 设置环境
        physicsClient = setup_environment()
        
        # 创建并加载机器人（先在高处加载，避免碰撞）
        robot = R1ProRobot(urdf_path, start_pos=[0, 0, 2.0], fixed_base=True)  # 先在高处加载
        robot.load()
        robot.print_info()
        
        stabilization_report = robot.initialize_for_grasp_annotations(
            ground_height=0.0,
            pose_profile="low_reach",
            lock_non_arm_joints=True,
            stiffness=120.0,
            settle_steps=120,
            lock_base=True
        )
        
        controllable_joints = stabilization_report.get("free", [])
        locked_joints = stabilization_report.get("locked", [])
        
        print("\n" + "=" * 60)
        print("关节锁定配置 - 只控制手臂和夹爪")
        print("=" * 60)
        
        if controllable_joints:
            print(f"\n可控制的关节（共 {len(controllable_joints)} 个）:")
            for joint_name in sorted(controllable_joints):
                print(f"  - {joint_name}")
        if locked_joints:
            print(f"\n已锁定的关节（共 {len(locked_joints)} 个）:")
            for joint_name in sorted(locked_joints):
                print(f"  - {joint_name}")
        
        print("=" * 60)
        
        # 创建控制滑块（使用初始位置作为滑块初始值）
        use_sliders = True  # 设置为 False 可以禁用滑块
        sliders = {}
        if use_sliders:
            # 确保所有关节都设置为初始位置（在创建滑块之前）
            robot.hold_initial_positions(stiffness=100.0)
            # 创建滑块，初始值会自动设置为当前关节位置
            sliders = create_joint_sliders(robot)
            # 再次确保所有关节都设置为初始位置
            robot.hold_initial_positions(stiffness=100.0)
            
            # 同步滑块值到当前关节位置（确保滑块值匹配当前位置）
            print("正在同步滑块值到当前关节位置...")
            current_positions = robot.get_joint_positions()
            for joint_name, current_pos in current_positions.items():
                if joint_name in robot.joint_name_to_id:
                    # 应用位置控制保持当前位置
                    robot.set_joint_control(joint_name, current_pos, p.POSITION_CONTROL, stiffness=100.0)
        
        print("\n" + "=" * 60)
        print("控制说明:")
        print("=" * 60)
        print("  鼠标操作:")
        print("    - 左键拖动: 旋转视角")
        print("    - 右键拖动: 平移视角")
        print("    - 滚轮: 缩放")
        print("  键盘操作:")
        print("    - 空格键: 暂停/继续仿真")
        print("    - ESC 键: 退出")
        if use_sliders:
            print("  滑块控制:")
            print("    - 使用右侧滑块调整关节位置")
        print("=" * 60)
        
        print("\n开始仿真...")
        print("按 ESC 键或关闭窗口退出\n")
        
        # 等待一下，确保 GUI 完全初始化（期间持续应用位置控制）
        print("等待 GUI 初始化...")
        for _ in range(10):
            robot.disable_default_joint_motors()
            # 持续应用位置控制，防止任何移动
            if use_sliders:
                # 使用滑块时，读取滑块值并应用
                for joint_name, slider_id in sliders.items():
                    try:
                        if slider_id is not None and slider_id >= 0:
                            target_pos = p.readUserDebugParameter(slider_id)
                            if target_pos is not None:
                                robot.set_joint_control(joint_name, target_pos, p.POSITION_CONTROL, stiffness=100.0)
                    except:
                        pass
            else:
                # 不使用滑块时，保持初始位置
                robot.hold_initial_positions(stiffness=100.0)
            
            p.stepSimulation()
            time.sleep(0.01)
        
        # 仿真循环
        step_count = 0
        use_stabilization = True  # 是否使用位置控制稳定关节
        
        # 记录上次的滑块值，用于检测变化
        last_slider_values = {}
        if use_sliders:
            for joint_name in sliders.keys():
                last_slider_values[joint_name] = None
        
        while True:
            robot.disable_default_joint_motors()
            # 首先保持所有锁定的关节在当前位置
            robot.hold_locked_joints(stiffness=100.0)
            
            # 更新关节控制 - 必须每个仿真步都应用
            if use_sliders:
                # 使用滑块控制时，读取滑块值并设置关节位置
                # 重要：必须持续应用位置控制，否则关节会因重力乱动
                for joint_name, slider_id in sliders.items():
                    # 跳过已锁定的关节（它们由 hold_locked_joints 处理）
                    if joint_name in robot.locked_joints:
                        continue
                    
                    try:
                        # 检查滑块 ID 是否有效（应该 >= 0）
                        if slider_id is not None and slider_id >= 0:
                            target_pos = p.readUserDebugParameter(slider_id)
                            if target_pos is not None:
                                # 持续应用位置控制（使用最大刚度保持位置）
                                # 使用 100% 的刚度，确保关节不会因重力移动
                                robot.set_joint_control(joint_name, target_pos, p.POSITION_CONTROL, stiffness=100.0)
                                last_slider_values[joint_name] = target_pos
                    except Exception as e:
                        # 如果读取失败，使用上次的值或初始位置
                        if joint_name in last_slider_values and last_slider_values[joint_name] is not None:
                            robot.set_joint_control(joint_name, last_slider_values[joint_name], p.POSITION_CONTROL, stiffness=100.0)
                        pass
            else:
                # 如果没有使用滑块，保持未锁定关节的初始位置（防止因重力乱动）
                if use_stabilization:
                    # 只保持未锁定关节的初始位置
                    for joint_name, target_position in robot.initial_joint_positions.items():
                        if joint_name not in robot.locked_joints:
                            robot.set_joint_control(joint_name, target_position, p.POSITION_CONTROL, stiffness=100.0)
            
            # 步进仿真
            p.stepSimulation()
            
            # 检查连接状态
            if not p.getConnectionInfo()["isConnected"]:
                break
            
            step_count += 1
            if step_count % 240 == 0:  # 每秒打印一次位置
                try:
                    pos, orn = p.getBasePositionAndOrientation(robot.robotId)
                    print(f"机器人位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                except:
                    pass
            
            time.sleep(1./240.)  # 240 Hz
            
    except KeyboardInterrupt:
        print("\n\n用户中断仿真")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n正在断开连接...")
        p.disconnect()
        print("✓ 已断开连接")


if __name__ == "__main__":
    main()
