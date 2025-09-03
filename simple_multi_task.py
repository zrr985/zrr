#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版多任务视觉检测系统
提供灵活的任务选择和控制
"""

import sys
import time
import threading
from multi_task_detection import MultiTaskDetectionSystem

# 导入配置文件
try:
    import detection_config as config
    print("✅ 简化控制界面成功加载配置文件")
except ImportError:
    print("⚠️ 配置文件不存在，使用默认配置")
    config = None

def print_menu():
    """打印菜单"""
    print("\n" + "="*50)
    print("🎯 多任务视觉检测系统控制台")
    print("="*50)
    print("可用任务:")
    print("1. infrared  - 红外入侵检测")
    print("2. flame     - 火焰检测")
    print("3. face      - 人脸识别")
    print("4. meter     - 仪表读数")
    print("5. hardhat   - 安全帽检测")
    print("6. smoking   - 吸烟检测")
    print("-" * 50)
    print("控制命令:")
    print("start <task>     - 启动指定任务 (如: start flame)")
    print("stop <task>      - 停止指定任务")
    print("start all        - 启动所有任务")
    print("stop all         - 停止所有任务")
    print("status           - 显示系统状态")
    print("help             - 显示此菜单")
    print("quit             - 退出程序")
    print("="*50)

def parse_command(command):
    """解析用户命令"""
    parts = command.strip().lower().split()
    if not parts:
        return None, None
    
    action = parts[0]
    target = parts[1] if len(parts) > 1 else None
    return action, target

def main():
    """主函数"""
    print("🌟 简化版多任务视觉检测系统")
    
    # 初始化配置
    if config is not None:
        try:
            if config.load_config():
                print("✅ 配置系统初始化成功")
            else:
                print("⚠️ 配置验证失败，使用默认配置")
        except Exception as e:
            print(f"⚠️ 配置初始化出错: {e}，使用默认配置")
    
    # 创建检测系统
    system = MultiTaskDetectionSystem()
    
    # 任务名称映射
    task_names = {
        '1': 'infrared', 'infrared': 'infrared',
        '2': 'flame', 'flame': 'flame', 
        '3': 'face', 'face': 'face',
        '4': 'meter', 'meter': 'meter',
        '5': 'hardhat', 'hardhat': 'hardhat',
        '6': 'smoking', 'smoking': 'smoking'
    }
    
    print_menu()
    
    try:
        while True:
            try:
                # 获取用户输入
                user_input = input("\n💬 请输入命令 (输入 'help' 查看帮助): ").strip()
                
                if not user_input:
                    continue
                
                action, target = parse_command(user_input)
                
                if action == 'help':
                    print_menu()
                
                elif action == 'quit' or action == 'exit':
                    print("👋 正在退出...")
                    break
                
                elif action == 'status':
                    system.print_status()
                
                elif action == 'start':
                    if target == 'all':
                        print("🚀 启动所有任务...")
                        system.start_all_tasks()
                    elif target in task_names:
                        task_name = task_names[target]
                        print(f"🚀 启动任务: {task_name}")
                        system.start_task(task_name)
                    else:
                        print("❌ 无效的任务名称，请输入 'help' 查看可用任务")
                
                elif action == 'stop':
                    if target == 'all':
                        print("🛑 停止所有任务...")
                        system.stop_all_tasks()
                    elif target in task_names:
                        task_name = task_names[target]
                        print(f"🛑 停止任务: {task_name}")
                        system.stop_task(task_name)
                    else:
                        print("❌ 无效的任务名称，请输入 'help' 查看可用任务")
                
                else:
                    print("❌ 无效的命令，请输入 'help' 查看可用命令")
            
            except KeyboardInterrupt:
                print("\n🛑 检测到 Ctrl+C，正在退出...")
                break
            except EOFError:
                print("\n🛑 输入结束，正在退出...")
                break
            except Exception as e:
                print(f"❌ 命令执行错误: {e}")
    
    finally:
        print("🧹 清理资源...")
        system.stop_all_tasks()
        print("✅ 程序已安全退出")

if __name__ == "__main__":
    main()
