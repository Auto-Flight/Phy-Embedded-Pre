#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from Dataset import get_dataloaders

# ================= 配置 =================
# 指向你的数据集路径
DATA_PATHS = ["./dataset_processed_05"] 
BATCH_SIZE = 16

def check_data_continuity(loader, name="Train"):
    print(f"\n{'='*20} Checking {name} Loader {'='*20}")
    
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Loader is empty!")
        return

    # [B, T=5, 14]
    pose_seq = batch['pose_seq']
    # [B, 10, 3]
    labels = batch['labels']
    
    # 提取 Batch 中所有样本的关键点
    # 1. 历史轨迹的最后一点 (相对于 t=0)
    # pose_seq: [B, T, 14] -> 取最后一帧 -> 取最后3维(dx,dy,dz) -> 取前3维
    # 注意：根据你的 Dataset, pose_seq 最后一维是 14: [10维Pose | 3维Traj | 1维Time]
    # 所以 3D轨迹是索引 10:13
    hist_last_pos = pose_seq[:, -1, 10:13] # [B, 3]
    
    # 验证：hist_last_pos 理论上应该非常接近 0 (因为输入是相对于当前帧 t=0 的)
    # 如果这里很大，说明 Input Alignment 算错了
    avg_hist_end_error = torch.mean(torch.norm(hist_last_pos, dim=1)).item()
    
    # 2. 未来轨迹的第一点 (相对于 t=0)
    # labels 是 delta，所以第一个绝对位置就是 labels[:, 0, :]
    fut_first_pos = labels[:, 0, :] # [B, 3]
    
    # 3. 计算 "跳变" (The Jump)
    # 物理上：人不可能在 0.1s 内瞬移。
    # 历史最后一点(t=0) 到 未来第一点(t=0.1) 的距离，应该等于 0.1s * 速度
    # 正常步行速度 ~1.0m/s -> 距离应约 0.1m
    # 如果距离过大 (> 0.5m) 或者方向反了，说明数据流断了
    
    # 因为 hist_last_pos 应该是0，我们直接看 fut_first_pos 的模长
    jump_dist = torch.norm(fut_first_pos - hist_last_pos, dim=1) # [B]
    avg_jump = torch.mean(jump_dist).item()
    max_jump = torch.max(jump_dist).item()
    
    print(f"1. Input History Check (t=0时刻归零检查):")
    print(f"   平均误差: {avg_hist_end_error:.6f} m (应 < 1e-5)")
    if avg_hist_end_error > 0.01:
        print("   [CRITICAL WARNING] 输入历史没有在 t=0 时刻归零！输入数据构建错误！")
    else:
        print("   [PASS] 输入历史锚点正确。")

    print(f"2. Continuity Check (连接处跳变检查):")
    print(f"   平均步长 (0.1s): {avg_jump:.4f} m")
    print(f"   最大步长 (0.1s): {max_jump:.4f} m")
    
    if avg_jump > 0.5:
        print("   [CRITICAL WARNING] 轨迹断裂！0.1s内移动了超过 0.5米，这不符合物理规律。")
        print("   可能原因：Labels 和 Inputs 来自不同的时间戳，或者索引错位。")
    elif avg_jump < 0.001:
        print("   [WARNING] 轨迹静止！目标几乎没动。检查是否加载了空数据。")
    else:
        print("   [PASS] 轨迹连接连贯。")

    # 4. 检查时间戳顺序 (Sequence Order)
    # 检查 pose_seq 的时间维 (最后一维) 是否是递增的
    times = pose_seq[0, :, 13].numpy()
    print(f"3. Time Sequence Check (Batch[0]):")
    print(f"   Values: {times}")
    is_increasing = np.all(np.diff(times) > 0)
    if not is_increasing:
        print("   [CRITICAL WARNING] 时间戳不是递增的！RNN 正在读取乱序数据！")
    else:
        print("   [PASS] 时间顺序正确。")

    # 5. 检查 Train vs Val 的数值分布差异
    # 计算 3D 轨迹的平均速度 (Z轴)
    # 输入轨迹的 Z 轴: index 12
    mean_z_vel = torch.mean(pose_seq[:, :, 12]).item()
    print(f"4. Distribution Check:")
    print(f"   平均 Z轴位移 (Input): {mean_z_vel:.4f}")
    
    return mean_z_vel

def main():
    # 强制使用 Time-Block Split (你的默认设置)
    train_loader, val_loader, _ = get_dataloaders(DATA_PATHS, batch_size=BATCH_SIZE, num_workers=0)
    
    print("\n" + "#"*50)
    print(" >>> DIAGNOSING TRAIN SET <<<")
    train_z = check_data_continuity(train_loader, "Train")
    
    print("\n" + "#"*50)
    print(" >>> DIAGNOSING VALIDATION SET <<<")
    val_z = check_data_continuity(val_loader, "Validation")
    
    print("\n" + "#"*50)
    print(" >>> SUMMARY <<<")
    print(f"Train Mean Z-Motion: {train_z:.4f}")
    print(f"Val   Mean Z-Motion: {val_z:.4f}")
    
    if abs(train_z - val_z) > 0.5: # 阈值可调
        print("\n[CONCLUSION] 检测到严重的分布偏移 (Distribution Shift)！")
        print("训练集和验证集的运动模式截然不同（例如一个向前走，一个向后走/静止）。")
        print("建议：尝试在 get_dataloaders 中使用 random.shuffle(sub_data) 暂时打乱时间顺序，")
        print("      看 Loss 是否能正常下降。如果打乱后正常，说明你需要更多样化的数据。")
    else:
        print("\n[CONCLUSION] 分布看似正常。如果以上 Check 都通过，请检查 Dropout 是否过高导致欠拟合，或学习率是否过大。")

if __name__ == "__main__":
    main()