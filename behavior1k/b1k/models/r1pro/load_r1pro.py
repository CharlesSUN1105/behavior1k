#!/usr/bin/env python3
"""
PyBullet 加载和显示 r1pro 机器人

使用方法:
    python load_r1pro.py
"""

import pybullet as p
import pybullet_data
import time
import os
import sys

# 添加当前目录到路径，以便导入 constants
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from constants import r1pro_init_joint_state
except ImportError:
    print("警告: 无法导入 constants.py，将使用默认关节状态")
    r1pro_init_joint_state = {}


def load_r1pro_robot():
    """加载 r1pro 机器人到 PyBullet 仿真环境"""
    
    # 1. 连接到 GUI 服务器
    print("正在连接到 PyBullet GUI...")
    physicsClient = p.connect(p.GUI)
    
    # 设置摄像头位置和朝向
    p.resetDebugVisualizerCamera(
        2.5,  # cameraDistance
        45,   # cameraYaw
        -30,  # cameraPitch
        [0, 0, 0.5]  # cameraTargetPosition
    )
    
    # 2. 设置搜索路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "r1pro_source.urdf")
    meshes_dir = os.path.join(current_dir, "meshes")
    
    print(f"URDF 路径: {urdf_path}")
    print(f"Meshes 目录: {meshes_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"错误: URDF 文件不存在: {urdf_path}")
        p.disconnect()
        return None
    
    # 3. 设置重力
    p.setGravity(0, 0, -9.81)
    
    # 4. 加载地面
    print("正在加载地面...")
    planeId = p.loadURDF("plane.urdf")
    
    # 5. 加载机器人
    print("正在加载 r1pro 机器人...")
    robot_start_pos = [0, 0, 0.5]  # 初始位置 (x, y, z)
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # 初始朝向
    
    # 加载 URDF，使用固定基座选项（可选）
    # flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
    robotId = p.loadURDF(
        urdf_path,
        robot_start_pos,
        robot_start_orientation,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
        useFixedBase=False  # 设置为 True 可以固定机器人基座
    )
    
    if robotId < 0:
        print("错误: 无法加载机器人 URDF")
        p.disconnect()
        return None
    
    print(f"✓ 成功加载机器人，ID: {robotId}")
    
    # 6. 获取机器人信息
    num_joints = p.getNumJoints(robotId)
    print(f"✓ 机器人关节数量: {num_joints}")
    
    # 打印所有关节信息
    print("\n关节信息:")
    joint_name_to_id = {}
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        joint_id = joint_info[0]
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        joint_name_to_id[joint_name] = i
        
        # 只打印可控制的关节（旋转关节或滑动关节）
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            print(f"  关节 {i}: {joint_name} (类型: {joint_type}, 范围: [{lower_limit:.2f}, {upper_limit:.2f}])")
    
    # 7. 设置初始关节状态（如果有的话）
    if r1pro_init_joint_state:
        print("\n正在设置初始关节状态...")
        for joint_name, joint_value in r1pro_init_joint_state.items():
            if joint_name in joint_name_to_id:
                joint_id = joint_name_to_id[joint_name]
                p.resetJointState(robotId, joint_id, joint_value)
                print(f"  ✓ {joint_name} = {joint_value:.3f} rad")
    
    # 8. 禁用默认的关节电机，让机器人受重力影响
    print("\n正在配置关节电机...")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        joint_type = joint_info[2]
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            # 设置关节为位置控制模式，但将力设为0（相当于无电机）
            p.setJointMotorControl2(
                robotId,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )
    
    print("\n✓ 机器人加载完成！")
    print("\n控制说明:")
    print("  - 鼠标左键拖动: 旋转视角")
    print("  - 鼠标右键拖动: 平移视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - 空格键: 暂停/继续仿真")
    print("  - ESC 键: 退出")
    
    return robotId, robot_start_pos, robot_start_orientation


def run_simulation(robotId, max_steps=None):
    """运行仿真循环"""
    print("\n开始仿真...")
    
    # 设置实时仿真（可选）
    # p.setRealTimeSimulation(1)
    
    step_count = 0
    while True:
        # 如果没有设置实时仿真，需要手动步进
        if not p.getRealTimeSimulation():
            p.stepSimulation()
            step_count += 1
            if max_steps and step_count >= max_steps:
                break
        
        # 检查连接状态
        if not p.getConnectionInfo()["isConnected"]:
            break
        
        time.sleep(1./240.)  # 约 240 Hz 的更新频率


def main():
    """主函数"""
    print("=" * 60)
    print("PyBullet r1pro 机器人加载器")
    print("=" * 60)
    
    try:
        # 加载机器人
        result = load_r1pro_robot()
        if result is None:
            return
        
        robotId, start_pos, start_orientation = result
        
        # 运行仿真
        print("\n按任意键开始仿真（或等待 3 秒自动开始）...")
        time.sleep(3)
        
        # 运行仿真（无限循环，直到用户关闭窗口）
        run_simulation(robotId)
        
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

