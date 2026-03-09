#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class DroneTrajectoryDataset(Dataset):
    """
    适配 Network 输入 (基于增量/Deltas):
    1. depth: 当前帧 [2, 64, 160] (RelDepth + Mask)
    2. state: 当前帧 [6] (vx, vy, vz, sin, cos, dist)
    3. traj_seq: 历史增量序列 [3, 4] (dx, dy, dz, dt) -> 时间正序 (Past -> Current)
    
    【意图 Label 说明】
    0: Straight (直行)
    1: Left (左转)
    2: Right (右转)
    """
    def __init__(self, data_list, augment=False, 
                 history_steps=5, future_steps=5, dt=0.3):  # <--- [修改] 默认值改为新参数，但最好由外部传入
        self.samples = data_list
        self.augment = augment
        
        # [修改] 保存配置参数
        self.history_steps = history_steps 
        self.future_steps = future_steps
        self.dt = dt
        
        self.IMG_W = 160.0
        self.IMG_H = 64.0

    def __len__(self):
        return len(self.samples)

    def apply_augmentation(self, depth, state, traj_seq, labels_delta, bbox, intention):
        # 1. 水平翻转 (Horizontal Flip)
        if random.random() < 0.5:
            # 图像翻转
            depth = torch.flip(depth, dims=[-1])
            bbox[0] = self.IMG_W - bbox[0]
            
            # 向量 X 轴取反 (增量和状态都需要)
            traj_seq[:, 0] *= -1     
            labels_delta[:, 0] *= -1 
            state[0] *= -1   # vx
            state[3] *= -1   # sin(yaw) -> x分量
            
            # 翻转意图标签: 1(Left) <-> 2(Right)
            if intention == 1:
                intention = torch.tensor(2, dtype=torch.long)
            elif intention == 2:
                intention = torch.tensor(1, dtype=torch.long)

        # 2. 比例缩放 (Scale)
        if random.random() < 0.4:
            scale = random.uniform(0.8, 1.2)
            # 增量可以直接缩放
            traj_seq[:, 0:3] *= scale
            labels_delta *= scale
            # 速度和深度也需要缩放
            state[0:3] *= scale 
            state[5] *= scale 

        # 3. 噪声
        if random.random() < 0.1:
            noise = torch.randn_like(depth) * 0.02
            depth = torch.clamp(depth + noise, 0.0, 1.0) 

        return depth, state, traj_seq, labels_delta, bbox, intention

    def __getitem__(self, idx):
        item_info = self.samples[idx]
        sample_root = item_info['dataset_root']
        curr_seq_id = str(item_info['seq_id']) 
        
        # --------------------------------------------------------
        # 1. 加载 Tensor 文件
        # --------------------------------------------------------
        d_path = os.path.join(sample_root, "depth_tensor", f"{int(curr_seq_id):06d}.npy")
        s_path = os.path.join(sample_root, "state_vector", f"{int(curr_seq_id):06d}.npy")
        
        # Depth
        try:
            depth = np.load(d_path).astype(np.float32)
            if depth.ndim == 3 and depth.shape[0] != 2: 
                depth = depth.transpose(2, 0, 1)
        except:
            depth = np.zeros((2, int(self.IMG_H), int(self.IMG_W)), dtype=np.float32)

        # State
        try:
            state = np.load(s_path).astype(np.float32)
        except:
            state = np.zeros((6,), dtype=np.float32)
            
        depth = torch.from_numpy(depth)
        state = torch.from_numpy(state)

        # --------------------------------------------------------
        # 2. [修改] 构建历史轨迹序列 - 使用 self.history_steps 和 self.dt
        # --------------------------------------------------------
        traj_deltas = np.array(item_info['history_traj'], dtype=np.float32)
        
        # 动态检查形状
        expected_shape = (self.history_steps, 3)
        if traj_deltas.shape != expected_shape:
            # 如果数据有问题，尝试 reshape 或者填充零
            if traj_deltas.size == self.history_steps * 3:
                traj_deltas = traj_deltas.reshape(expected_shape)
            else:
                # print(f"Warning: Seq {curr_seq_id} history shape mismatch {traj_deltas.shape} vs {expected_shape}")
                traj_deltas = np.zeros(expected_shape, dtype=np.float32)
            
        # [修改] 使用正确的 dt
        time_vals = np.full((self.history_steps, 1), self.dt, dtype=np.float32) 
        traj_seq_np = np.concatenate([traj_deltas, time_vals], axis=1) # [5, 4]
        traj_seq = torch.from_numpy(traj_seq_np)

        # --------------------------------------------------------
        # 3. [修改] 标签 - 使用 self.future_steps
        # --------------------------------------------------------
        labels_delta_np = np.array(item_info['labels_delta'], dtype=np.float32)
        
        # 动态 Reshape
        if labels_delta_np.size == self.future_steps * 3:
            labels_delta = torch.tensor(labels_delta_np.reshape(self.future_steps, 3), dtype=torch.float32)
        else:
            labels_delta = torch.zeros((self.future_steps, 3), dtype=torch.float32)
            
        bbox = torch.tensor(item_info['bbox'], dtype=torch.float32)

        raw_intent = item_info['intention_label']
        if raw_intent > 2: raw_intent = 0 
        intention = torch.tensor(raw_intent, dtype=torch.long)

        if self.augment:
            depth, state, traj_seq, labels_delta, bbox, intention = self.apply_augmentation(
                depth, state, traj_seq, labels_delta, bbox, intention
            )

        return {
            'depth': depth,               
            'state': state,               
            'traj_seq': traj_seq,         
            'labels_delta': labels_delta, 
            'intention': intention,       
            'bbox': bbox,
            'seq_id': item_info['seq_id']
        }


def get_dataloaders(dataset_dirs, batch_size=32, num_workers=4, split_ratio=[0.85, 0.14, 0.01], seed=42, history_steps=5, future_steps=5, dt=0.3):  # split_ratio=[0.8, 0.01, 0.19]  [0.9, 0.09, 0.01]
    if isinstance(dataset_dirs, str):
        dataset_dirs = [dataset_dirs]
        
    all_data = []
    print(f"Loading data from {len(dataset_dirs)} directories...")
    print(f"Config: History={history_steps}, Future={future_steps}, dt={dt}")
    
    random.seed(seed)
    # ... (加载 JSON 部分保持不变) ...
    for root_path in dataset_dirs:
        json_path = os.path.join(root_path, "labels.json")
        if not os.path.exists(json_path):
            continue 
        with open(json_path, 'r') as f:
            sub_data = json.load(f)
        for sample in sub_data:
            sample['dataset_root'] = root_path
        all_data.extend(sub_data)
        print(f" -> {os.path.basename(root_path)}: {len(sub_data)} samples")

    random.shuffle(all_data)
    n_total = len(all_data)
    if n_total == 0: raise ValueError("No data found!")

    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    test_data = all_data[n_train + n_val :]
    train_data = all_data[:n_train]
    val_data = all_data[n_train : n_train + n_val]
    
    # [修改] 实例化时传入参数
    train_ds = DroneTrajectoryDataset(train_data, augment=True, history_steps=history_steps, future_steps=future_steps, dt=dt) 
    val_ds = DroneTrajectoryDataset(val_data, augment=False, history_steps=history_steps, future_steps=future_steps, dt=dt)    
    test_ds = DroneTrajectoryDataset(test_data, augment=False, history_steps=history_steps, future_steps=future_steps, dt=dt)  
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# =========================================================================
#  批量可视化保存脚本
# =========================================================================
if __name__ == "__main__":
    import sys
    
    # DEBUG_DATA_PATHS = [
    #     "./dataset_processed_01", 
    #     "./dataset_processed_02", 
    #     "./dataset_processed_03", 
    #     "./dataset_processed_04", 
    #     "./dataset_processed_05", 
    #     "./dataset_processed_06", 
    #     "./dataset_processed_07", 
    #     "./dataset_processed_08", 
    #     "./dataset_processed_09", 
    #     "./dataset_processed_10", 
    # ]  
    DEBUG_DATA_PATHS = [
        "./Data_1.5s/dataset_processed_01", 
        "./Data_1.5s/dataset_processed_02", 
        "./Data_1.5s/dataset_processed_03", 
        "./Data_1.5s/dataset_processed_04", 
        "./Data_1.5s/dataset_processed_05", 
        "./Data_1.5s/dataset_processed_06", 
        "./Data_1.5s/dataset_processed_07", 
        "./Data_1.5s/dataset_processed_08", 
        "./Data_1.5s/dataset_processed_09", 
        "./Data_1.5s/dataset_processed_10", 
    ] 
    SAVE_DIR = "./debug_vis_results"              
    MAX_SAMPLES = 50                        
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    try:
        if not os.path.exists(DEBUG_DATA_PATHS[0]):
             print(f"Dataset path not found, please check: {DEBUG_DATA_PATHS[0]}")
             
        train_loader, _, _ = get_dataloaders(DEBUG_DATA_PATHS, batch_size=1, num_workers=0, history_steps=5, future_steps=5, dt=0.3)
    except Exception as e:
        print(f"DataLoader init failed: {e}")
        sys.exit(0)
    
    intent_map = {0: "Straight", 1: "Left", 2: "Right"}
    
    print(f"=== Starting Batch Debug ===")
    count = 0
    
    for i, batch in enumerate(tqdm(train_loader, total=MAX_SAMPLES)):
        if count >= MAX_SAMPLES:
            break
            
        try:
            depth = batch['depth']          
            state = batch['state']          
            traj_seq = batch['traj_seq']    
            labels = batch['labels_delta']  
            intent = batch['intention']     
            seq_id = batch['seq_id'][0]

            depth_np = depth[0].numpy()
            state_np = state[0].numpy()
            hist_deltas = traj_seq[0].numpy()[:, 0:3] 
            fut_deltas = labels[0].numpy()
            
            intent_val = intent.item()
            intent_str = intent_map.get(intent_val, f"Unknown({intent_val})")
            
            # state: [vx, vy, vz, sin, cos, dist]
            vx, vy, vz, sin_y, cos_y, dist_m = state_np
            
            # -------------------------------------------------
            # 绘图逻辑 
            # -------------------------------------------------
            fig = plt.figure(figsize=(14, 5))
            ax1 = fig.add_subplot(1, 3, 1)

            # 轨迹重建
            rev_deltas = hist_deltas[::-1] 
            rev_pos = np.cumsum(rev_deltas, axis=0)
            hist_pos_rel = -rev_pos 
            full_hist_pos = np.vstack([hist_pos_rel[::-1], [0,0,0]])
            
            fut_pos_rel = np.cumsum(fut_deltas, axis=0)
            full_fut_pos = np.vstack([[0,0,0], fut_pos_rel])

            ax1.plot(full_hist_pos[:, 0], full_hist_pos[:, 2], 'b--o', label="Past", alpha=0.7)
            ax1.plot(full_fut_pos[:, 0], full_fut_pos[:, 2], 'r-x', label="Future", linewidth=2)
            ax1.scatter(0, 0, s=150, c='k', marker='*', label="Current")
            
            # 辅助箭头 1: 速度 (Velocity)
            ax1.arrow(0, 0, vx * 0.5, vz * 0.5, head_width=0.05, fc='g', ec='g', label="Vel")
            
            # === [修复] 辅助箭头 2: 机头朝向 (Heading) ===
            # 使用 sin_y (x轴分量) 和 cos_y (z轴分量)
            ax1.arrow(0, 0, sin_y * 0.5, cos_y * 0.5, head_width=0.05, fc='m', ec='m', label="Head")
            
            ax1.set_title(f"Seq: {seq_id} | {intent_str}\nVel:({vx:.2f}, {vz:.2f})")
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Z [m]")
            ax1.grid(True)
            ax1.legend()
            ax1.axis('equal') 

            # Depth
            rel_depth = depth_np[0]
            ax2 = fig.add_subplot(1, 3, 2)
            im2 = ax2.imshow(rel_depth, cmap='jet', vmin=-1.0, vmax=1.0)
            ax2.set_title(f"RelDepth (D:{dist_m:.2f}m)")
            plt.colorbar(im2, ax=ax2)

            # Mask
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(depth_np[1], cmap='gray')
            ax3.set_title("Mask")

            plt.tight_layout()
            
            filename = f"vis_{i:03d}_seq{seq_id}_{intent_str}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            plt.savefig(save_path)
            plt.close()
            
            count += 1
            
        except Exception as e:
            print(f"Error batch {i}: {e}")
            plt.close()
            continue
            
    print(f"\nDone! Saved {count} images to {SAVE_DIR}")