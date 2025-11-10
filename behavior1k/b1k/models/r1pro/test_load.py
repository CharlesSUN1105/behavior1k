#!/usr/bin/env python3
"""
简单的 r1pro 机器人加载测试脚本

用于快速验证 URDF 文件是否可以正确加载
"""

import pybullet as p
import pybullet_data
import os
import sys

def test_load_urdf():
    """测试加载 URDF 文件"""
    print("=" * 60)
    print("R1Pro URDF 加载测试")
    print("=" * 60)
    
    # 获取文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "r1pro_source.urdf")
    meshes_dir = os.path.join(current_dir, "meshes")
    
    print(f"\n检查文件:")
    print(f"  URDF 路径: {urdf_path}")
    print(f"  Meshes 目录: {meshes_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"  ❌ URDF 文件不存在!")
        return False
    else:
        print(f"  ✓ URDF 文件存在")
    
    if not os.path.exists(meshes_dir):
        print(f"  ❌ Meshes 目录不存在!")
        return False
    else:
        mesh_files = [f for f in os.listdir(meshes_dir) if f.endswith('.obj')]
        print(f"  ✓ Meshes 目录存在 ({len(mesh_files)} 个 OBJ 文件)")
    
    # 连接到 PyBullet（DIRECT 模式，不显示 GUI）
    print(f"\n连接到 PyBullet (DIRECT 模式)...")
    physicsClient = p.connect(p.DIRECT)
    
    try:
        # 设置搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载地面
        print("加载地面...")
        planeId = p.loadURDF("plane.urdf")
        print(f"  ✓ 地面加载成功 (ID: {planeId})")
        
        # 加载机器人
        print(f"\n加载机器人 URDF...")
        robot_start_pos = [0, 0, 0.5]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        robotId = p.loadURDF(
            urdf_path,
            robot_start_pos,
            robot_start_orientation,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase=False
        )
        
        if robotId < 0:
            print(f"  ❌ 无法加载机器人 URDF")
            return False
        
        print(f"  ✓ 机器人加载成功 (ID: {robotId})")
        
        # 获取机器人信息
        num_joints = p.getNumJoints(robotId)
        print(f"\n机器人信息:")
        print(f"  关节数量: {num_joints}")
        
        # 统计可控制关节
        controllable_count = 0
        joint_names = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robotId, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8')
            joint_names.append(joint_name)
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                controllable_count += 1
        
        print(f"  可控制关节数: {controllable_count}")
        print(f"  总链接数: {num_joints + 1}")  # +1 为 base_link
        
        # 获取基座信息
        base_pos, base_orn = p.getBasePositionAndOrientation(robotId)
        print(f"\n基座信息:")
        print(f"  位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
        print(f"  朝向: [{base_orn[0]:.3f}, {base_orn[1]:.3f}, {base_orn[2]:.3f}, {base_orn[3]:.3f}]")
        
        # 运行几步仿真测试
        print(f"\n运行仿真测试 (10 步)...")
        p.setGravity(0, 0, -9.81)
        
        for i in range(10):
            p.stepSimulation()
        
        final_pos, final_orn = p.getBasePositionAndOrientation(robotId)
        print(f"  初始位置: [{robot_start_pos[0]:.3f}, {robot_start_pos[1]:.3f}, {robot_start_pos[2]:.3f}]")
        print(f"  最终位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        
        if abs(final_pos[2] - robot_start_pos[2]) > 0.01:
            print(f"  ✓ 物理仿真正常（机器人受重力影响）")
        else:
            print(f"  ⚠ 机器人位置未变化（可能基座被固定）")
        
        print(f"\n" + "=" * 60)
        print("✓ 所有测试通过！URDF 文件可以正常加载")
        print("=" * 60)
        print(f"\n可以运行以下命令启动 GUI:")
        print(f"  python load_r1pro.py")
        print(f"  或")
        print(f"  python load_r1pro_advanced.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        p.disconnect()
        print(f"\n已断开连接")


if __name__ == "__main__":
    success = test_load_urdf()
    sys.exit(0 if success else 1)

