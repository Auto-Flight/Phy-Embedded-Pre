#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import matplotlib.gridspec as gridspec

# 请确保 Dataset.py 在同一目录或 Python 路径下
from Dataset import get_dataloaders

CONF_PRED_HORIZON = 5 # 预测步长 (对应 future_steps, 原代码混用3或5)
CONF_HIST_HORIZON = 5 # 历史步长 (对应 history_steps)
CONF_DT = 0.3 # 采样时间间隔 (秒)
CONF_NUM_INTENTIONS = 3 # 意图类别数


CONF_MAX_STEP_DIST = 1.0 # 单步最大位移 (m) -> 用于反归一化 scale           1.0
CONF_MAX_VEL = 5.0 # 最大速度 (m/s)
CONF_MAX_DEPTH = 10.0 # 最大深度 (m)
# =========================================================================
# 1. 损失函数定义 (Robust Physical NLL)
# =========================================================================
class RobustPhysicalNLLLoss(nn.Module):
    """
    鲁棒的物理 NLL 损失函数
    """
    def __init__(self, 
                 fixed_scale=CONF_MAX_STEP_DIST,      # 反归一化系数 (max_step_dist)
                 max_diff_thresh=2.0,  # [防御] 脏数据阈值(米)
                 num_steps=CONF_PRED_HORIZON,          # 预测步长
                 decay_rate=0.8,       # 时间衰减 (越远越不重要)
                 dim_weights=[1.0, 1.0, 1.0] # [x, y, z]
                 ):
        super(RobustPhysicalNLLLoss, self).__init__()
        
        self.nll_loss = nn.GaussianNLLLoss(reduction='none')
        self.fixed_scale = fixed_scale
        self.max_diff_thresh = max_diff_thresh
        
        # 使用 register_buffer 注册张量，自动处理 device 转移
        time_w = [decay_rate ** i for i in range(num_steps)]
        self.register_buffer('time_weights', torch.tensor(time_w, dtype=torch.float32).view(1, num_steps, 1))
        self.register_buffer('dim_weights', torch.tensor(dim_weights, dtype=torch.float32).view(1, 1, 3))

    def forward(self, pred_mu_norm, pred_sigma_norm, target_deltas_real):
        # 1. 反归一化到物理空间
        pred_mu_real = pred_mu_norm * self.fixed_scale
        pred_sigma_real = pred_sigma_norm * self.fixed_scale
        pred_var_real = pred_sigma_real ** 2
        
        # 2. 脏数据检测 (Masking)
        with torch.no_grad():
            diff = torch.norm(pred_mu_real - target_deltas_real, p=2, dim=-1) # [B, T]
            valid_mask = (diff < self.max_diff_thresh).float().unsqueeze(-1) # [B, T, 1]

        # 3. 计算基础 NLL
        raw_nll = self.nll_loss(pred_mu_real, target_deltas_real, pred_var_real) # [B, T, 3]
        
        # 4. 应用加权
        weighted_nll = raw_nll * self.time_weights * self.dim_weights
        
        # 5. 应用 Mask 并求平均
        masked_loss_sum = (weighted_nll * valid_mask).sum()
        valid_elements = valid_mask.sum() * 3.0 + 1e-6
        
        return masked_loss_sum / valid_elements

# =========================================================================
# 2. 可视化工具 (更新意图映射)
# =========================================================================
class TrajectoryVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 意图映射
        self.intent_map = {0: "Straight", 1: "Left", 2: "Right"}

    def plot_sample(self, sample_idx, gt_deltas, pred_deltas, hist_deltas, 
                    gt_intent, pred_intent, depth_img, state_vec):
        """
        IROS 风格紧凑可视化 (V8 - 最终版: 密度可调 + 全局平滑 + 关键点突出)
        """
        from matplotlib.ticker import MaxNLocator # 确保引入刻度定位器

        # ================= [参数调试区] =================
        GRID_DENSITY = 14       # [调试] 网格密度: 数字越大网格越密 (建议 6-12)
        PT_SIZE_FUTURE = 60    # [调试] 未来轨迹点的大小
        LINE_WIDTH = 2.5       # [调试] 平滑曲线的线宽
        # ===============================================
        
        # --- A. 数据解析 ---
        vx, vy, vz, sin_y, cos_y, dist_m = state_vec
        
        # 1. 轨迹还原
        origin = np.zeros((1, 3))
        # GT Future
        gt_fut_pos = np.vstack([origin, np.cumsum(gt_deltas, axis=0)])
        # Pred Future
        pred_fut_pos = np.vstack([origin, np.cumsum(pred_deltas, axis=0)])
        
        # History
        if hist_deltas is not None:
            rev_deltas = hist_deltas[::-1] 
            rev_pos = np.cumsum(rev_deltas, axis=0)
            hist_pos_rel = -rev_pos 
            full_hist_pos = np.vstack([hist_pos_rel[::-1], origin]) 
        else:
            full_hist_pos = origin

        # 文字内容
        gt_str = self.intent_map.get(gt_intent, "Unk")
        pred_str = self.intent_map.get(pred_intent, "Unk")
        intent_text = f"Intent: {gt_str} (GT) $\\rightarrow$ {pred_str} (Pred)"
        
        # --- B. 绘图设置 ---
        fig = plt.figure(figsize=(6, 8)) 
        
        target_hspace = 0.05 
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.8], hspace=target_hspace)
        
        # === 上部分: 深度图 ===
        ax_depth = fig.add_subplot(gs[0])
        h_img, w_img = depth_img.shape
        ax_depth.imshow(depth_img, cmap='jet', vmin=-1.0, vmax=1.0, aspect='auto')
        
        # Dist 标记
        ax_depth.text(w_img / 2, h_img / 1.3, f"Dist: {dist_m:.1f}m", 
                      color='white', fontsize=15, fontweight='bold',
                      ha='center', va='center',
                      bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
        
        ax_depth.set_xticks([])
        ax_depth.set_yticks([])

        # === 下部分: 轨迹图 ===
        ax_traj = fig.add_subplot(gs[1])
        
        # 辅助函数: 绘制平滑曲线
        def plot_smooth_curve(ax, points, color, label_line, label_pt, marker, line_style='-'):
            if len(points) < 2: return
            
            # 1. 绘制高亮关键点 (突出)
            ax.scatter(points[:, 0], points[:, 2], c=color, s=PT_SIZE_FUTURE, 
                       marker=marker, edgecolors='white', linewidth=0.8, 
                       zorder=6, label=label_pt)
            
            # 2. 拟合平滑曲线
            t_steps = np.arange(len(points))
            # 2阶或3阶拟合 (点太少就用线性)
            deg = 2 if len(points) > 2 else 1
            coeff_x = np.polyfit(t_steps, points[:, 0], deg)
            coeff_z = np.polyfit(t_steps, points[:, 2], deg)
            
            t_smooth = np.linspace(0, len(points)-1, 50)
            x_smooth = np.polyval(coeff_x, t_smooth)
            z_smooth = np.polyval(coeff_z, t_smooth)
            
            ax.plot(x_smooth, z_smooth, color=color, linestyle=line_style, 
                    linewidth=LINE_WIDTH, alpha=0.9, zorder=5, label=label_line)

        # 1. 历史轨迹 (History)
        history_color = 'royalblue'
        if len(full_hist_pos) > 0:
            # 真实点
            ax_traj.scatter(full_hist_pos[:, 0], full_hist_pos[:, 2], 
                            c='navy', s=40, alpha=1.0, zorder=5, label='History (Pts)')
            # 拟合曲线 (虚线)
            if len(full_hist_pos) > 2:
                t_steps = np.arange(len(full_hist_pos))
                coeff_x = np.polyfit(t_steps, full_hist_pos[:, 0], 2)
                coeff_z = np.polyfit(t_steps, full_hist_pos[:, 2], 2)
                t_smooth = np.linspace(0, len(full_hist_pos)-1, 50)
                x_smooth = np.polyval(coeff_x, t_smooth)
                z_smooth = np.polyval(coeff_z, t_smooth)
                
                ax_traj.plot(x_smooth, z_smooth, color=history_color, linestyle='--', 
                             linewidth=2, alpha=0.6, label='_nolegend_')
            else:
                ax_traj.plot(full_hist_pos[:, 0], full_hist_pos[:, 2], color=history_color, 
                             linestyle='--', alpha=0.6)

        # 2. GT Future (红色, 圆点, 实线)
        plot_smooth_curve(ax_traj, gt_fut_pos, color='#E62E2D', 
                          label_line='_nolegend_', label_pt='GT Future', 
                          marker='o')

        # 3. Pred Future (橙色, 三角, 实线)
        plot_smooth_curve(ax_traj, pred_fut_pos, color='#FF8C00', 
                          label_line='_nolegend_', label_pt='Pred Future', 
                          marker='^')
        
        # 3. 当前位置 (Current)
        ax_traj.scatter(0, 0, c='k', s=220, marker='*', label='Current', zorder=10)
        
        # 4. 速度向量 & Heading
        vec_scale = 0.25 
        # 使用 annotate 绘制箭头
        ax_traj.annotate('', 
                         xy=(vx * vec_scale, vz * vec_scale), 
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=history_color, 
                                         linestyle='--', linewidth=2, 
                                         shrinkA=0, shrinkB=0, mutation_scale=20),
                         zorder=8)
        # [关键] 添加一个看不到的 Dummy Line，仅为了让 Legend 显示标签
        ax_traj.plot([], [], color=history_color, linestyle='--', linewidth=2, label="Vel Vector")
        
        heading_scale = 0.25
        # 使用 annotate 绘制箭头
        ax_traj.annotate('', 
                         xy=(sin_y * heading_scale, cos_y * heading_scale), 
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='magenta', 
                                         linestyle='--', linewidth=2, 
                                         shrinkA=0, shrinkB=0, mutation_scale=20),
                         zorder=7)
        # [关键] 添加 Dummy Line 用于图例
        ax_traj.plot([], [], color='magenta', linestyle='--', linewidth=2, label="Heading")

        # 5. Intent 标签
        ax_traj.text(0.5, 1.0, intent_text, transform=ax_traj.transAxes, 
                     ha='center', va='top', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9, zorder=20))

        # 6. 动态 Legend
        all_pts = np.vstack([full_hist_pos, gt_fut_pos, pred_fut_pos])
        avg_x = np.mean(all_pts[:, 0])
        legend_loc = 'lower right' if avg_x > 0 else 'lower left'
        ax_traj.legend(loc=legend_loc, fontsize=12, framealpha=0.9, fancybox=True)

        # 7. 动态坐标轴范围
        min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
        min_z, max_z = all_pts[:, 2].min(), all_pts[:, 2].max()
        margin = 0.3
        
        span_x = max_x - min_x
        span_z = max_z - min_z
        max_span = max(span_x, span_z)
        center_x = (max_x + min_x) / 2
        center_z = (max_z + min_z) / 2
        
        final_span = max_span + margin
        ax_traj.set_xlim(center_x - final_span/2, center_x + final_span/2)
        ax_traj.set_ylim(center_z - final_span/2, center_z + final_span/2)
        
        # === 核心修改: 密度可调的网格 + 无边框 ===
        
        # 隐藏边框
        for spine in ax_traj.spines.values():
            spine.set_visible(False)
            
        # 隐藏刻度显示
        ax_traj.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
        
        # 轴标签
        ax_traj.set_xlabel("Lateral X (m)", fontsize=10)
        ax_traj.set_ylabel("Forward Z (m)", fontsize=10)

        # [核心] 设置网格密度
        ax_traj.xaxis.set_major_locator(MaxNLocator(nbins=GRID_DENSITY))
        ax_traj.yaxis.set_major_locator(MaxNLocator(nbins=GRID_DENSITY))

        # 强制显示网格
        ax_traj.grid(True, linestyle='-', color='#B0B0B0', linewidth=1.2, alpha=0.6, zorder=0)
        
        ax_traj.set_aspect('equal', adjustable='box')

        # 保存
        filename = f"vis_paper_{sample_idx:04d}_{gt_str}_{pred_str}.jpg"
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

# =========================================================================
# 3. 网络
# =========================================================================
# （1） 传统卷积 + MLP 拼接 (Model_Baseline_MLP)
class Model_Baseline_MLP(nn.Module):
    """
    对照组 1: 不使用神经卡尔曼，而是使用传统的 Concat + MLP 融合
    """
    def __init__(self, pred_horizon=CONF_PRED_HORIZON, num_intentions=CONF_NUM_INTENTIONS, hidden_dim=256,
                 max_step_dist=CONF_MAX_STEP_DIST, max_vel=CONF_MAX_VEL, max_depth=CONF_MAX_DEPTH):
        super(Model_Baseline_MLP, self).__init__()
        self.pred_horizon = pred_horizon
        self.register_buffer('norm_step', torch.tensor(max_step_dist))
        self.register_buffer('norm_vel', torch.tensor(max_vel))
        self.register_buffer('norm_depth', torch.tensor(max_depth))

        # --- A. 视觉 (保持不变) ---
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.visual_fc = nn.Sequential(nn.Conv2d(64, 16, 1), nn.Flatten(), nn.Linear(640, 128), nn.ReLU())
        self.obstacle_module = SceneAwareObstacleModule(in_channels=64, num_bins=5, feature_h=4)

        # --- B. 状态 (保持不变) ---
        self.vel_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.heading_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())

        # --- C. 历史 (保持不变) ---
        self.traj_embed = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.traj_gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # --- D. 融合层 (修改点：替换 NKF 为 MLP) ---
        # 总输入维度 = Traj(64) + Vel(32) + Vis(128) + Head(32) + Obs(5) = 261
        total_in_dim = 64 + 32 + 128 + 32 + 5
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # 传统 MLP 需要 Dropout 防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # --- E. Heads ---
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, pred_horizon * 3 * 2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_intentions))

    def normalize_data(self, state, traj_seq):
        norm_state = state.clone()
        norm_state[:, :3] /= self.norm_vel
        norm_state[:, 5] /= self.norm_depth
        norm_traj = traj_seq.clone()
        norm_traj[:, :, :3] /= self.norm_step 
        return norm_state, norm_traj

    def forward(self, depth, state, traj_seq):
        batch_size = depth.size(0)
        s_norm, t_norm = self.normalize_data(state, traj_seq)
        
        # 特征提取
        t_emb = self.traj_embed(t_norm); _, h_traj = self.traj_gru(t_emb)
        feat_traj = h_traj[-1]
        feat_vel = self.vel_encoder(s_norm[:, :3])
        feat_head = self.heading_encoder(s_norm[:, 3:])
        
        cnn_feat = self.depth_cnn(depth)
        feat_vis = self.visual_fc(cnn_feat)
        feat_obs = self.obstacle_module(cnn_feat)

        # [修改点] 简单拼接
        concat_feat = torch.cat([feat_traj, feat_vel, feat_vis, feat_head, feat_obs], dim=1)
        
        # [修改点] MLP 融合
        fused_emb = self.fusion_mlp(concat_feat)

        # 输出
        raw_params = self.regressor(fused_emb).view(batch_size, self.pred_horizon, 3, 2)
        mu = raw_params[..., 0]
        sigma = F.softplus(raw_params[..., 1]) + 0.01
        intent = self.classifier(fused_emb)
        return mu, sigma, intent

    def inference(self, depth, state, traj_seq, sample_mode='best'):
        self.eval()
        with torch.no_grad():
            mu, sigma, intent_logits = self.forward(depth, state, traj_seq)
            pred_intent = torch.argmax(intent_logits, dim=1)
            norm_out = mu if sample_mode == 'best' else mu + sigma * torch.randn_like(mu)
            return norm_out * self.norm_step, pred_intent

# =========================================================================
# （2） SAOM & NKF
# =========================================================================
class SceneAwareObstacleModule(nn.Module):
    def __init__(self, in_channels, num_bins=5, feature_h=4):
        """
        feature_h: 输入特征图的高度 (根据 backbone 输出确定, 这里假设是 4)
        """
        super(SceneAwareObstacleModule, self).__init__()
        
        # 1. 特征提取 (保持不变)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        
        # 2. [关键修改] 垂直压缩层 (Vertical Compression)
        # 替换原本的 MaxPool。
        # kernel_size=(feature_h, 1) 表示在高度上全覆盖，在宽度上只看1个像素
        # 这样网络就能学会：哪一行的高度才是真正的障碍物。
        self.vertical_compress = nn.Conv2d(16, 16, kernel_size=(feature_h, 1), bias=False)
        
        # 3. 风险映射 (保持不变)
        self.risk_mapper = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # 4. 水平方向离散化 (保持不变)
        self.bin_pool = nn.AdaptiveAvgPool2d((1, num_bins))

    def forward(self, x):
        # x: [B, 64, 4, 10]
        
        # 1. 降维 -> [B, 16, 4, 10]
        feat = self.channel_att(x)
        
        # 2. [关键修改] 垂直压缩
        # 不再是粗暴的 Max，而是加权求和。
        # 如果训练得当，网络会学会给底部的 feat 赋予极小的权重。
        # Output: [B, 16, 1, 10]
        feat_h = self.vertical_compress(feat) 
        
        # 3. 映射为风险图 -> [B, 1, 1, 10]
        risk_map = self.risk_mapper(feat_h)
        
        # 4. 离散化 -> [B, 5]
        obstacle_vec = self.bin_pool(risk_map).view(x.size(0), -1) 
        
        return obstacle_vec

# class NeuralKalmanFusion(nn.Module):
#     def __init__(self, motion_dim, obs_dim, hidden_dim):
#         super(NeuralKalmanFusion, self).__init__()
#         self.gate_net = nn.Sequential(
#             nn.Linear(motion_dim + obs_dim, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim // 2, hidden_dim), 
#             nn.Sigmoid() 
#         )
#         self.motion_proj = nn.Linear(motion_dim, hidden_dim)
#         self.obs_proj = nn.Linear(obs_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, h_motion, h_obs):
#         m_emb = self.motion_proj(h_motion)
#         o_emb = self.obs_proj(h_obs)
#         concat_feat = torch.cat([h_motion, h_obs], dim=1)
#         K = self.gate_net(concat_feat) 
#         h_fused = (1 - K) * m_emb + K * o_emb
#         return self.norm(h_fused), K
    
class NeuralKalmanFusion(nn.Module):
    def __init__(self, motion_dim, obs_dim, hidden_dim):
        super(NeuralKalmanFusion, self).__init__()
        
        # --- 修改点 1: 激活函数换成 LeakyReLU ---
        # LeakyReLU(0.1) 允许负区间的梯度回传，防止神经元“脑死亡”
        self.gate_net = nn.Sequential(
            nn.Linear(motion_dim + obs_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),  # [Change] ReLU -> LeakyReLU
            nn.Linear(hidden_dim // 2, hidden_dim), 
            nn.Sigmoid() 
        )
        
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # --- 修改点 2: 偏置初始化 (Bias Trick) ---
        # 手动初始化 Gate Net 最后一层的 bias
        # 目的：让初始 K 值偏离 0.5，打破对称性，制造“梯度势能”
        self._init_gate_bias()

    def _init_gate_bias(self):
        # gate_net 的结构是: [Linear(0), LeakyReLU(1), Linear(2), Sigmoid(3)]
        last_linear = self.gate_net[2]
        
        # 将 bias 设为 -2.0
        # 效果：Sigmoid(-2.0) ≈ 0.12
        # 含义：初始状态下，网络会 88% 信任 Motion，12% 信任 Observation。
        # 逻辑：通常运动预测在短时间内是可靠的，强迫网络只有在视觉特征非常有说服力时，才去提升 K 值。
        nn.init.constant_(last_linear.bias, -2.0)
        
        # (可选) 同时让该层的权重初始化得小一点，确保训练初期 bias 起主导作用
        nn.init.xavier_uniform_(last_linear.weight, gain=0.01)

    def forward(self, h_motion, h_obs):
        m_emb = self.motion_proj(h_motion)
        o_emb = self.obs_proj(h_obs)
        concat_feat = torch.cat([h_motion, h_obs], dim=1)
        
        K = self.gate_net(concat_feat) 
        
        # 卡尔曼更新公式：(1-K)*Prior + K*Measurement
        h_fused = (1 - K) * m_emb + K * o_emb
        
        return self.norm(h_fused), K

class ContextAwareTrajectoryNet(nn.Module):
    def __init__(self,
        pred_horizon=CONF_PRED_HORIZON, # [CONF]
        num_intentions=CONF_NUM_INTENTIONS, # [CONF]
        hidden_dim=256,
        max_step_dist=CONF_MAX_STEP_DIST, # [CONF]
        max_vel=CONF_MAX_VEL, # [CONF]
        max_depth=CONF_MAX_DEPTH # [CONF]
        ):
        super(ContextAwareTrajectoryNet, self).__init__()
        self.pred_horizon = pred_horizon
        self.num_intentions = num_intentions
        
        self.register_buffer('norm_step', torch.tensor(max_step_dist))
        self.register_buffer('norm_vel', torch.tensor(max_vel))
        self.register_buffer('norm_depth', torch.tensor(max_depth))

        # --- A. 视觉感知分支 ---
        self.depth_cnn_backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        
        self.obstacle_module = SceneAwareObstacleModule(in_channels=64, num_bins=5)
        
        self.visual_fc = nn.Sequential(
            nn.Conv2d(64, 16, 1), 
            nn.Flatten(),         
            nn.Linear(640, 128),
            nn.ReLU(inplace=True)
        )

        # --- B. 状态解耦 ---
        self.vel_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.heading_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())

        # --- C. 历史轨迹 ---
        self.traj_embed = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.traj_gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # --- D. 神经卡尔曼融合 ---
        # Motion: 64(GRU) + 32(Vel) = 96
        # Obs: 128(Vis) + 32(Head) + 5(ObsVec) = 165
        self.nkf_fusion = NeuralKalmanFusion(
            motion_dim=96, 
            obs_dim=165, 
            hidden_dim=hidden_dim
        )

        # --- E. Heads ---
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pred_horizon * 3 * 2) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_intentions) # 输出维度现在是3
        )

    def normalize_data(self, state, traj_seq):
        norm_state = state.clone()
        norm_state[:, :3] /= self.norm_vel
        norm_state[:, 5] /= self.norm_depth
        norm_traj = traj_seq.clone()
        norm_traj[:, :, :3] /= self.norm_step 
        return norm_state, norm_traj

    def forward(self, depth, state, traj_seq):
        batch_size = depth.size(0)
        s_norm, t_norm = self.normalize_data(state, traj_seq)
        
        velocity = s_norm[:, :3]      
        heading_info = s_norm[:, 3:]  

        # Part 1: Motion Prior
        t_emb = self.traj_embed(t_norm)
        _, h_traj = self.traj_gru(t_emb)
        feat_traj = h_traj[-1] 
        feat_vel = self.vel_encoder(velocity)
        h_motion = torch.cat([feat_traj, feat_vel], dim=1) 

        # Part 2: Observation
        cnn_feat_map = self.depth_cnn_backbone(depth) 
        obstacle_vec = self.obstacle_module(cnn_feat_map)
        feat_visual = self.visual_fc(cnn_feat_map)
        feat_heading = self.heading_encoder(heading_info)
        h_obs = torch.cat([feat_visual, feat_heading, obstacle_vec], dim=1) 

        # Part 3: Fusion
        fused_emb, K_gain = self.nkf_fusion(h_motion, h_obs)

        # Part 4: Output
        raw_params = self.regressor(fused_emb).view(batch_size, self.pred_horizon, 3, 2)
        pred_mu_norm = raw_params[..., 0] 
        pred_sigma_norm = F.softplus(raw_params[..., 1]) + 0.01 
        pred_intent = self.classifier(fused_emb)

        return pred_mu_norm, pred_sigma_norm, pred_intent

    def inference(self, depth, state, traj_seq, sample_mode='best'):
        self.eval()
        with torch.no_grad():
            mu, sigma, intent_logits = self.forward(depth, state, traj_seq)
            pred_intent = torch.argmax(intent_logits, dim=1)
            if sample_mode == 'best': norm_out = mu
            else: norm_out = mu + sigma * torch.randn_like(mu)
            real_deltas = norm_out * self.norm_step
            return real_deltas, pred_intent

# （3） 无 Heading (Model_NoHeading)
class Model_NoHeading(nn.Module):
    """
    消融实验 2: 使用 NKF 架构，但移除 Heading 输入
    [参数平衡] hidden_dim 调大至 264，以补偿移除 Heading Encoder 带来的参数减少
    """
    def __init__(self,
        pred_horizon=CONF_PRED_HORIZON, # [CONF]
        num_intentions=CONF_NUM_INTENTIONS, # [CONF]
        hidden_dim=264, # <--- [修改] 微调至 264 (原256)
        max_step_dist=CONF_MAX_STEP_DIST, # [CONF]
        max_vel=CONF_MAX_VEL, # [CONF]
        max_depth=CONF_MAX_DEPTH # [CONF]
        ):
        super(Model_NoHeading, self).__init__()
        self.pred_horizon = pred_horizon
        self.register_buffer('norm_step', torch.tensor(max_step_dist))
        self.register_buffer('norm_vel', torch.tensor(max_vel))
        self.register_buffer('norm_depth', torch.tensor(max_depth))

        # 1. 视觉与障碍物 (保持不变)
        self.depth_cnn_backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.obstacle_module = SceneAwareObstacleModule(64, 5, 4)
        self.visual_fc = nn.Sequential(nn.Conv2d(64, 16, 1), nn.Flatten(), nn.Linear(640, 128), nn.ReLU())

        # 2. 状态 (移除 Heading)
        self.vel_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        # self.heading_encoder 被移除

        # 3. 历史
        self.traj_embed = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.traj_gru = nn.GRU(32, 64, 1, batch_first=True)

        # 4. NKF 融合
        # Motion: 64 + 32 = 96
        # Obs: 128 + 5 = 133 (无 Heading)
        self.nkf_fusion = NeuralKalmanFusion(motion_dim=96, obs_dim=133, hidden_dim=hidden_dim)

        # 5. Heads (输入维度随 hidden_dim 变大)
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, pred_horizon * 3 * 2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_intentions))

    def normalize_data(self, state, traj_seq):
        norm_state = state.clone(); norm_state[:, :3] /= self.norm_vel; norm_state[:, 5] /= self.norm_depth
        norm_traj = traj_seq.clone(); norm_traj[:, :, :3] /= self.norm_step 
        return norm_state, norm_traj

    def forward(self, depth, state, traj_seq):
        batch_size = depth.size(0)
        s_norm, t_norm = self.normalize_data(state, traj_seq)
        
        t_emb = self.traj_embed(t_norm); _, h_traj = self.traj_gru(t_emb)
        feat_traj = h_traj[-1]
        feat_vel = self.vel_encoder(s_norm[:, :3])
        h_motion = torch.cat([feat_traj, feat_vel], dim=1)

        cnn_feat = self.depth_cnn_backbone(depth)
        feat_vis = self.visual_fc(cnn_feat)
        feat_obs = self.obstacle_module(cnn_feat)
        h_obs = torch.cat([feat_vis, feat_obs], dim=1) # 无 Heading

        fused_emb, _ = self.nkf_fusion(h_motion, h_obs)
        
        raw_params = self.regressor(fused_emb).view(batch_size, self.pred_horizon, 3, 2)
        mu = raw_params[..., 0]; sigma = F.softplus(raw_params[..., 1]) + 0.01
        intent = self.classifier(fused_emb)
        return mu, sigma, intent
    
    def inference(self, d, s, t, sample_mode='best'):
        self.eval()
        with torch.no_grad():
            mu, sigma, intent = self.forward(d, s, t)
            pred_int = torch.argmax(intent, dim=1)
            norm_out = mu if sample_mode == 'best' else mu + sigma * torch.randn_like(mu)
            return norm_out * self.norm_step, pred_int
        
# （4） 无深度输入 (Model_NoDepth)
class Model_NoDepth(nn.Module):
    """
    消融实验 3: 使用 NKF 架构，但移除 Visual/Depth 输入
    [精准参数控制版] 
    目标：4.3 MB
    策略：hidden_dim 保持 256，仅通过调节 param_compensator 的宽度来对齐参数量。
    """
    def __init__(self,
        pred_horizon=CONF_PRED_HORIZON, # [CONF]
        num_intentions=CONF_NUM_INTENTIONS, # [CONF]
        hidden_dim=256, # <--- [关键] 恢复为 256，防止核心网络过大
        max_step_dist=CONF_MAX_STEP_DIST, # [CONF]
        max_vel=CONF_MAX_VEL, # [CONF]
        max_depth=CONF_MAX_DEPTH # [CONF]
        ):
        super(Model_NoDepth, self).__init__()
        self.pred_horizon = pred_horizon
        self.register_buffer('norm_step', torch.tensor(max_step_dist))
        self.register_buffer('norm_vel', torch.tensor(max_vel))
        
        # [移除] depth_cnn, obstacle_module, visual_fc

        # 状态
        self.vel_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.heading_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())

        # 历史
        self.traj_embed = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.traj_gru = nn.GRU(32, 64, 1, batch_first=True)

        # NKF 融合
        # Motion: 96, Obs: 32 (仅 Heading)
        self.nkf_fusion = NeuralKalmanFusion(motion_dim=96, obs_dim=32, hidden_dim=hidden_dim)

        # [参数补偿模块] 
        # 我们需要填补移除 CNN 造成的空缺。
        # 保持 hidden_dim=256 的情况下，调节 expansion_dim 可以线性增加参数。
        # 设为 1700 大约增加 0.9M 参数 (约 3.5MB)，加上基础部分，总共约 4.3MB。
        expansion_dim = 400  # <--- [可调] 调节这个数字来精确命中 4.3MB
        
        self.param_compensator = nn.Sequential(
            nn.Linear(hidden_dim, expansion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(expansion_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Heads
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, pred_horizon * 3 * 2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_intentions))

    def normalize_data(self, state, traj_seq):
        norm_state = state.clone(); norm_state[:, :3] /= self.norm_vel
        norm_traj = traj_seq.clone(); norm_traj[:, :, :3] /= self.norm_step 
        return norm_state, norm_traj

    def forward(self, depth, state, traj_seq):
        batch_size = state.size(0)
        s_norm, t_norm = self.normalize_data(state, traj_seq)
        
        t_emb = self.traj_embed(t_norm); _, h_traj = self.traj_gru(t_emb)
        feat_traj = h_traj[-1]
        feat_vel = self.vel_encoder(s_norm[:, :3])
        h_motion = torch.cat([feat_traj, feat_vel], dim=1)

        feat_head = self.heading_encoder(s_norm[:, 3:])
        h_obs = feat_head

        fused_emb, _ = self.nkf_fusion(h_motion, h_obs)

        # 补偿参数量
        fused_emb = self.param_compensator(fused_emb)
        
        raw_params = self.regressor(fused_emb).view(batch_size, self.pred_horizon, 3, 2)
        mu = raw_params[..., 0]; sigma = F.softplus(raw_params[..., 1]) + 0.01
        intent = self.classifier(fused_emb)
        return mu, sigma, intent
    
    def inference(self, d, s, t, sample_mode='best'):
        self.eval()
        with torch.no_grad():
            mu, sigma, intent = self.forward(d, s, t)
            pred_int = torch.argmax(intent, dim=1)
            norm_out = mu if sample_mode == 'best' else mu + sigma * torch.randn_like(mu)
            return norm_out * self.norm_step, pred_int
        
# =========================================================================
# （5） 无 SAOM 模块 (Model_NoSAOM)
# =========================================================================
class Model_NoSAOM(nn.Module):
    """
    消融实验 4: 移除 SAOM 模块，验证显式障碍物建模的必要性。
    
    [参数平衡策略]
    1. 移除 SAOM (约减少 3k 参数)。
    2. NKF 的输入维度 obs_dim 从 165 降为 160 (Vis 128 + Head 32)。
    3. 补偿：将 hidden_dim 从 256 增加到 260。
       这会在 NKF 内部矩阵（投影层、门控层）以及后续的 Regressor/Classifier 中
       增加约 10k+ 参数，足以覆盖移除 SAOM 带来的参数损失，确保模型容量只增不减。
    """
    def __init__(self, 
             pred_horizon=CONF_PRED_HORIZON,      # [CONF]
             num_intentions=CONF_NUM_INTENTIONS,  # [CONF]
             hidden_dim=260,  # <--- [修改] 从 256 增加到 260 以补偿参数
             max_step_dist=CONF_MAX_STEP_DIST,    # [CONF]
             max_vel=CONF_MAX_VEL,                # [CONF]
             max_depth=CONF_MAX_DEPTH             # [CONF]
             ):
        super(Model_NoSAOM, self).__init__()
        self.pred_horizon = pred_horizon
        self.register_buffer('norm_step', torch.tensor(max_step_dist))
        self.register_buffer('norm_vel', torch.tensor(max_vel))
        self.register_buffer('norm_depth', torch.tensor(max_depth))

        # --- A. 视觉感知分支 (保持不变) ---
        self.depth_cnn_backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        
        # [修改] 移除了 self.obstacle_module = SceneAwareObstacleModule(...)
        
        self.visual_fc = nn.Sequential(
            nn.Conv2d(64, 16, 1), 
            nn.Flatten(),         
            nn.Linear(640, 128),
            nn.ReLU(inplace=True)
        )

        # --- B. 状态解耦 (保持不变) ---
        self.vel_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.heading_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())

        # --- C. 历史轨迹 (保持不变) ---
        self.traj_embed = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.traj_gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # --- D. 神经卡尔曼融合 ---
        # Motion: 64(GRU) + 32(Vel) = 96 (不变)
        # Obs: 128(Vis) + 32(Head) = 160 (减少了 SAOM 的 5 维)
        self.nkf_fusion = NeuralKalmanFusion(
            motion_dim=96, 
            obs_dim=160,     # <--- [修改] 维度变小
            hidden_dim=hidden_dim
        )

        # --- E. Heads (输入维度随 hidden_dim 略微增大) ---
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pred_horizon * 3 * 2) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_intentions)
        )

    def normalize_data(self, state, traj_seq):
        norm_state = state.clone()
        norm_state[:, :3] /= self.norm_vel
        norm_state[:, 5] /= self.norm_depth
        norm_traj = traj_seq.clone()
        norm_traj[:, :, :3] /= self.norm_step 
        return norm_state, norm_traj

    def forward(self, depth, state, traj_seq):
        batch_size = depth.size(0)
        s_norm, t_norm = self.normalize_data(state, traj_seq)
        
        velocity = s_norm[:, :3]      
        heading_info = s_norm[:, 3:]  

        # Part 1: Motion Prior
        t_emb = self.traj_embed(t_norm)
        _, h_traj = self.traj_gru(t_emb)
        feat_traj = h_traj[-1] 
        feat_vel = self.vel_encoder(velocity)
        h_motion = torch.cat([feat_traj, feat_vel], dim=1) 

        # Part 2: Observation (无 SAOM)
        cnn_feat_map = self.depth_cnn_backbone(depth) 
        
        # [修改] 不再调用 obstacle_module
        feat_visual = self.visual_fc(cnn_feat_map)
        feat_heading = self.heading_encoder(heading_info)
        
        # [修改] 拼接时移除 obstacle_vec
        h_obs = torch.cat([feat_visual, feat_heading], dim=1) 

        # Part 3: Fusion
        fused_emb, K_gain = self.nkf_fusion(h_motion, h_obs)

        # Part 4: Output
        raw_params = self.regressor(fused_emb).view(batch_size, self.pred_horizon, 3, 2)
        pred_mu_norm = raw_params[..., 0] 
        pred_sigma_norm = F.softplus(raw_params[..., 1]) + 0.01 
        pred_intent = self.classifier(fused_emb)

        return pred_mu_norm, pred_sigma_norm, pred_intent

    def inference(self, depth, state, traj_seq, sample_mode='best'):
        self.eval()
        with torch.no_grad():
            mu, sigma, intent_logits = self.forward(depth, state, traj_seq)
            pred_intent = torch.argmax(intent_logits, dim=1)
            norm_out = mu if sample_mode == 'best' else mu + sigma * torch.randn_like(mu)
            return norm_out * self.norm_step, pred_intent

# =========================================================================
# （6） 无 SAOM和Heading 模块
# =========================================================================     
class Model_NoSAOM_NoHeading(nn.Module):
    """
    消融实验 5: 双重消融 - 同时移除 SAOM 和 Heading 输入。
    
    [参数平衡策略]
    1. 移除 SAOM (减小) 和 Heading Encoder (减小)。
    2. NKF 的输入维度 obs_dim 从 160 (Vis+Head) 降为 128 (仅 Vis)。
       这会导致 NKF 的 H_net 和 K_net 权重矩阵大幅缩小。
    3. 补偿：将 hidden_dim 从 260 进一步增加到 270。
       这显著增加了 GRU 隐层、NKF 内部矩阵以及 Regressor/Classifier 第一层的参数量，
       确保模型总参数量不低于 Baseline，维持强大的拟合能力。
    """
    def __init__(self, 
             pred_horizon=CONF_PRED_HORIZON,      # [CONF]
             num_intentions=CONF_NUM_INTENTIONS,  # [CONF]
             hidden_dim=270,  # <--- [修改] 从 260 增加到 270 以补偿双重移除的参数
             max_step_dist=CONF_MAX_STEP_DIST,    # [CONF]
             max_vel=CONF_MAX_VEL,                # [CONF]
             max_depth=CONF_MAX_DEPTH             # [CONF]
             ):
        super(Model_NoSAOM_NoHeading, self).__init__()
        self.pred_horizon = pred_horizon
        self.register_buffer('norm_step', torch.tensor(max_step_dist))
        self.register_buffer('norm_vel', torch.tensor(max_vel))
        self.register_buffer('norm_depth', torch.tensor(max_depth))

        # --- A. 视觉感知分支 (保持不变) ---
        self.depth_cnn_backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        
        # [修改] 移除了 self.obstacle_module
        
        self.visual_fc = nn.Sequential(
            nn.Conv2d(64, 16, 1), 
            nn.Flatten(),         
            nn.Linear(640, 128),
            nn.ReLU(inplace=True)
        )

        # --- B. 状态解耦 (移除 Heading Encoder) ---
        self.vel_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        # [修改] 移除了 self.heading_encoder = ...

        # --- C. 历史轨迹 (保持不变) ---
        self.traj_embed = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.traj_gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # --- D. 神经卡尔曼融合 ---
        # Motion: 64(GRU) + 32(Vel) = 96 (不变)
        # Obs: 仅 128(Vis) = 128 (移除了 SAOM 和 Heading)
        self.nkf_fusion = NeuralKalmanFusion(
            motion_dim=96, 
            obs_dim=128,     # <--- [修改] 维度降至 128
            hidden_dim=hidden_dim
        )

        # --- E. Heads (输入维度随 hidden_dim 增大) ---
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pred_horizon * 3 * 2) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_intentions)
        )

    def normalize_data(self, state, traj_seq):
        norm_state = state.clone()
        norm_state[:, :3] /= self.norm_vel
        # 注意：这里保留原索引 5 用于深度，假设输入数据结构未变，只是我们在 forward 中不使用朝向数据
        norm_state[:, 5] /= self.norm_depth 
        norm_traj = traj_seq.clone()
        norm_traj[:, :, :3] /= self.norm_step 
        return norm_state, norm_traj

    def forward(self, depth, state, traj_seq):
        #print("Size:", depth.size(), state.size(), traj_seq.size())
        batch_size = depth.size(0)
        s_norm, t_norm = self.normalize_data(state, traj_seq)
        
        # 1. 提取速度 (忽略朝向信息)
        velocity = s_norm[:, :3]      
        # [修改] 不再提取 heading_info

        # Part 1: Motion Prior
        t_emb = self.traj_embed(t_norm)
        _, h_traj = self.traj_gru(t_emb)
        feat_traj = h_traj[-1] 
        feat_vel = self.vel_encoder(velocity)
        h_motion = torch.cat([feat_traj, feat_vel], dim=1) 

        # Part 2: Observation (无 SAOM, 无 Heading)
        cnn_feat_map = self.depth_cnn_backbone(depth) 
        feat_visual = self.visual_fc(cnn_feat_map)
        
        # [修改] h_obs 仅由视觉特征构成
        h_obs = feat_visual 

        # Part 3: Fusion
        fused_emb, K_gain = self.nkf_fusion(h_motion, h_obs)

        # Part 4: Output
        raw_params = self.regressor(fused_emb).view(batch_size, self.pred_horizon, 3, 2)
        pred_mu_norm = raw_params[..., 0] 
        pred_sigma_norm = F.softplus(raw_params[..., 1]) + 0.01 
        pred_intent = self.classifier(fused_emb)

        return pred_mu_norm, pred_sigma_norm, pred_intent

    def inference(self, depth, state, traj_seq, sample_mode='best'):
        self.eval()
        with torch.no_grad():
            mu, sigma, intent_logits = self.forward(depth, state, traj_seq)
            pred_intent = torch.argmax(intent_logits, dim=1)
            norm_out = mu if sample_mode == 'best' else mu + sigma * torch.randn_like(mu)
            return norm_out * self.norm_step, pred_intent
        
# =========================================================================
# 4. 训练系统
# =========================================================================
class TrajectoryPredictorSystem:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. 初始化模型 
        if args.exp == 'baseline_mlp':
            print(">>> Loading Model: Baseline (MLP Concat)")
            model_class = Model_Baseline_MLP
        elif args.exp == 'no_heading':
            print(">>> Loading Model: Ablation (No Heading)")
            model_class = Model_NoHeading
        elif args.exp == 'no_depth':
            print(">>> Loading Model: Ablation (No Depth)")
            model_class = Model_NoDepth
        elif args.exp == 'no_SAOM':
            print(">>> Loading Model: Ablation (No SAOM)")
            model_class = Model_NoSAOM
        elif args.exp == 'no_SAOM_heading':
            print(">>> Loading Model: Ablation (No SAOM_heading)")
            model_class = Model_NoSAOM_NoHeading
        else:
            print(">>> Loading Model: Ours (NKF + Full Features)")
            model_class = ContextAwareTrajectoryNet # 你的原版模型类名

        # [CONF] 实例化模型时传入全局配置
        self.model = model_class(
            pred_horizon=CONF_PRED_HORIZON, 
            num_intentions=CONF_NUM_INTENTIONS,     
            max_step_dist=CONF_MAX_STEP_DIST,
            max_vel=CONF_MAX_VEL,
            max_depth=CONF_MAX_DEPTH
        ).to(self.device)
        
        model_scale = self.model.norm_step.item()

        # 2. 初始化 Loss 
        self.reg_criterion = RobustPhysicalNLLLoss(
            fixed_scale=model_scale,
            max_diff_thresh=2.0,         
            num_steps=CONF_PRED_HORIZON,
            decay_rate=1.0,  # 0.4              
            dim_weights=[1.0, 0.1, 1.0]  
        ).to(self.device) 
        
        self.cls_criterion = nn.CrossEntropyLoss().to(self.device)

        # 3. 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=5e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        os.makedirs(args.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')
        
        self.lambda_reg = 1.0
        self.lambda_cls = 1.3

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            print(f"Loading {path}...")
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print(f"Checkpoint not found: {path}")

    def run_train(self, train_loader, val_loader):
        print("Start Training...")
        # 初始化日志字典
        loss_hist = {'train': [], 'val': [], 'reg': [], 'cls': []}

        # ================= [新增] 早停参数定义 =================
        patience = 15       # 连续多少个epoch不下降则停止
        trigger_times = 0   # 当前连续未下降计数器
        # ========================================================

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            ep_loss, ep_reg, ep_cls = 0., 0., 0.
            
            pbar = tqdm(train_loader, desc=f"Ep {epoch}/{self.args.epochs}")
            for batch in pbar:
                # ... (数据加载和前向传播代码保持不变) ...
                depth = batch['depth'].to(self.device)
                state = batch['state'].to(self.device)
                traj = batch['traj_seq'].to(self.device)
                gt_deltas = batch['labels_delta'].to(self.device)
                gt_intent = batch['intention'].to(self.device)

                mu, sigma, intent_logits = self.model(depth, state, traj)

                loss_reg = self.reg_criterion(mu, sigma, gt_deltas)
                loss_cls = self.cls_criterion(intent_logits, gt_intent)
                total = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls

                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()

                ep_loss += total.item()
                ep_reg += loss_reg.item()
                ep_cls += loss_cls.item()
                pbar.set_postfix({'Tot': f"{total.item():.3f}", 'Reg': f"{loss_reg.item():.3f}"})

            # Validation
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            # 记录本 Epoch 数据
            avg_train = ep_loss / len(train_loader)
            loss_hist['train'].append(avg_train)
            loss_hist['val'].append(val_loss)
            loss_hist['reg'].append(ep_reg / len(train_loader))
            loss_hist['cls'].append(ep_cls / len(train_loader))

            print(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: {val_loss:.4f}")
            
            # ================= [修改] 保存最佳模型 + 早停逻辑 =================
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # 如果loss下降了，重置计数器
                trigger_times = 0
                
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss},
                           os.path.join(self.args.save_dir, "best_model.pth"))
                # print("Best model saved.") # 可选：打印保存信息
            else:
                # 如果loss没有下降，计数器+1
                trigger_times += 1
                print(f"EarlyStopping counter: {trigger_times} out of {patience}")
                
                # 检查是否达到阈值
                if trigger_times >= patience:
                    print(f"Early stopping triggered! Val loss has not improved for {patience} epochs.")
                    break # 跳出 for epoch 循环
            # =================================================================
        
        # === [新增] 保存 Loss 日志为 JSON 文件 ===
        # 这样未来你就可以读取不同实验的 log.json 来画对比图了
        log_path = os.path.join(self.args.save_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(loss_hist, f, indent=4)
        print(f"Training Log saved to: {log_path}")

        # 绘图 (保持不变)
        self.plot_curves(loss_hist)

    def validate(self, loader):
        self.model.eval()
        total = 0.
        with torch.no_grad():
            for batch in loader:
                d = batch['depth'].to(self.device)
                s = batch['state'].to(self.device)
                t = batch['traj_seq'].to(self.device)
                gt_d = batch['labels_delta'].to(self.device)
                gt_i = batch['intention'].to(self.device)
                
                mu, sigma, logits = self.model(d, s, t)
                l_r = self.reg_criterion(mu, sigma, gt_d)
                l_c = self.cls_criterion(logits, gt_i)
                total += (self.lambda_reg * l_r + self.lambda_cls * l_c).item()
        return total / len(loader)
    
    def calculate_performance_metrics(self, loader):
        """
        计算定量性能指标: 
        1. ADE (Average Displacement Error) - 全过程平均
        2. IDE (Initial Displacement Error) - 起步误差 (t=0), 原 First DE
        3. FDE (Final Displacement Error)   - 终点误差 (t=T), 业界标准 FDE
        4. Intent Acc - 意图准确率
        5. Avg GT Speed - 平均速度
        """
        self.model.eval()
        
        total_ade = 0.0
        total_ide = 0.0  # [改名] Initial Displacement Error (t=0)
        total_fde = 0.0  # [标准] Final Displacement Error (t=T)
        correct_intent = 0
        total_samples = 0
        total_gt_speed = 0.0 
        
        DT = CONF_DT
        
        print(">>> Calculating Performance Metrics...")
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Metrics"):
                # 1. 加载数据
                d = batch['depth'].to(self.device)
                s = batch['state'].to(self.device)
                t = batch['traj_seq'].to(self.device)
                gt_deltas = batch['labels_delta'].to(self.device)
                gt_intent = batch['intention'].to(self.device)
                
                batch_size = d.size(0)
                total_samples += batch_size
                
                # 2. 推理
                mu, _, intent_logits = self.model(d, s, t)
                
                # --- A. 意图指标 ---
                pred_intent = torch.argmax(intent_logits, dim=1)
                correct_intent += (pred_intent == gt_intent).sum().item()
                
                # --- B. 轨迹指标 ---
                # 反归一化并转为绝对位置
                pred_deltas_real = mu * self.model.norm_step
                pred_pos = torch.cumsum(pred_deltas_real, dim=1) 
                gt_pos = torch.cumsum(gt_deltas, dim=1)          
                
                # 计算欧氏距离 [B, T]
                l2_dist = torch.norm(pred_pos - gt_pos, p=2, dim=-1)
                
                # 1. ADE (全时段平均)
                total_ade += l2_dist.mean().item() * batch_size
                
                # 2. IDE (Initial Displacement Error, t=0, 反应起始精度)
                total_ide += l2_dist[:, 0].mean().item() * batch_size
                
                # 3. FDE (Final Displacement Error, t=-1, 终点预测精度)
                total_fde += l2_dist[:, -1].mean().item() * batch_size

                # --- C. 统计速度 ---
                step_dist = torch.norm(gt_deltas, p=2, dim=-1)
                avg_speed_sample = step_dist.mean(dim=1) / DT
                total_gt_speed += avg_speed_sample.sum().item()

        
        # 平均化
        avg_ade = total_ade / total_samples
        avg_ide = total_ide / total_samples # 起步误差
        avg_fde = total_fde / total_samples # 终点误差
        acc_intent = correct_intent / total_samples * 100.0
        avg_dataset_speed = total_gt_speed / total_samples
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6 
        
        print("\n" + "="*50)
        print(f"   Performance Metrics (N={total_samples})")
        print("="*50)
        print(f"1. Trajectory Precision:")
        print(f"   - ADE (Mean):       {avg_ade:.4f} m")
        print(f"   - IDE (Initial):    {avg_ide:.4f} m  <-- (t=0, Reaction)")
        print(f"   - FDE (Final):      {avg_fde:.4f} m  <-- (t=End, Long-term)")
        print("-" * 50)
        print(f"2. Semantic Understanding:")
        print(f"   - Intent Accuracy:  {acc_intent:.2f} %")
        print("-" * 50)
        print(f"3. Dataset Statistics:")
        print(f"   - Avg GT Speed:     {avg_dataset_speed:.2f} m/s")
        print("-" * 50)
        print(f"4. Efficiency:")
        print(f"   - Params:           {total_params:.2f} M")
        print("="*50 + "\n")
        
        # 返回三个主要指标
        return avg_ade, avg_ide, avg_fde, acc_intent
    
    def plot_curves(self, hist):
        """
        绘制高颜值训练曲线 (双Y轴版)
        """
        train_loss = hist['train']
        val_loss = hist['val']
        reg_loss = hist['reg']
        cls_loss = hist['cls']
        epochs = range(1, len(train_loss) + 1)
        
        # ==========================================
        # 图表 1: 总损失 (保持不变)
        # ==========================================
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
        plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
        
        best_idx = np.argmin(val_loss)
        best_val_loss = val_loss[best_idx]
        best_epoch = best_idx + 1
        plt.scatter(best_epoch, best_val_loss, s=150, c='red', marker='*', zorder=10,
                    label=f'Best Val: {best_val_loss:.4f} @ Ep {best_epoch}')
        
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Total Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "loss_curve_total.png"), dpi=120)
        plt.close()

        # ==========================================
        # 图表 2: 分量损失 (双 Y 轴优化版)
        # ==========================================
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- 左轴: Regression Loss (NLL) ---
        color_reg = 'tab:green'
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Regression Loss (NLL)', color=color_reg, fontsize=12)
        # 绘制曲线
        l1 = ax1.plot(epochs, reg_loss, color=color_reg, linewidth=2, label='Regression Loss (NLL)')
        # 设置刻度颜色
        ax1.tick_params(axis='y', labelcolor=color_reg)
        # 添加网格 (以左轴为准)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- 右轴: Classification Loss (CE) ---
        ax2 = ax1.twinx()  # 实例化共享X轴的第二个坐标轴
        color_cls = 'tab:purple'
        ax2.set_ylabel('Classification Loss (CE)', color=color_cls, fontsize=12)
        # 绘制曲线
        l2 = ax2.plot(epochs, cls_loss, color=color_cls, linewidth=2, label='Classification Loss (CE)')
        # 设置刻度颜色
        ax2.tick_params(axis='y', labelcolor=color_cls)

        # --- 合并图例 ---
        # 因为有两个轴，需要手动收集图例句柄
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', fontsize=11, ncol=2) # 图例放中间上方

        plt.title('Loss Components Analysis (Dual Axis)', fontsize=14)
        plt.tight_layout()
        
        save_path_comp = os.path.join(self.args.save_dir, "loss_curve_components.png")
        plt.savefig(save_path_comp, dpi=120)
        plt.close()
        print(f"Saved optimized curves to {self.args.save_dir}")

    def run_test(self, loader, visualize=False):
        self.model.eval()
        
        # 1. 首先计算硬指标
        self.calculate_performance_metrics(loader)
        
        # 2. 如果需要，再进行可视化
        if visualize:
            vis = TrajectoryVisualizer(os.path.join(self.args.save_dir, "vis_test_results"))
            print(f"Starting visualization...")
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(loader, desc="Visualizing")):
                    # ... (原有的可视化代码保持不变) ...
                    d = batch['depth'].to(self.device)
                    s = batch['state'].to(self.device)
                    t = batch['traj_seq'].to(self.device)
                    gt_d = batch['labels_delta'].to(self.device)
                    gt_i = batch['intention'].to(self.device)
                    
                    pred_deltas, pred_intent = self.model.inference(d, s, t, sample_mode='best')
                    
                    if i % 5 == 0:
                        # 数据转换
                        hist_deltas = t[0, :, :3].cpu().numpy()
                        depth_img = d[0, 0].cpu().numpy()
                        state_vec = s[0].cpu().numpy()
                        gt_deltas_np = gt_d[0].cpu().numpy()
                        pred_deltas_np = pred_deltas[0].cpu().numpy()
                        gt_intent_val = gt_i[0].item()
                        pred_intent_val = pred_intent[0].item()
                        
                        vis.plot_sample(i, gt_deltas_np, pred_deltas_np, hist_deltas,
                                        gt_intent_val, pred_intent_val, depth_img, state_vec)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--data_dirs', nargs='+', required=True, help='Dataset directories')
    parser.add_argument('--save_dir', default='./checkpoints_nll')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    # [新增] 实验类型参数
    parser.add_argument('--exp', type=str, default='ours', 
                        choices=['ours', 'baseline_mlp', 'no_heading', 'no_depth', 'no_SAOM', 'no_SAOM_heading'],
                        help='Experiment setting: ours | baseline_mlp | no_heading | no_depth | no_SAOM')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(
    args.data_dirs, 
    batch_size=args.batch_size, 
    num_workers=0,
    history_steps=CONF_HIST_HORIZON, 
    future_steps=CONF_PRED_HORIZON,   
    dt=CONF_DT
)
    system = TrajectoryPredictorSystem(args)
    
    if args.mode == 'train':
        if args.checkpoint: system.load_checkpoint(args.checkpoint)
        system.run_train(train_loader, val_loader)
    else:
        ckpt = args.checkpoint if args.checkpoint else os.path.join(args.save_dir, "best_model.pth")
        if not os.path.exists(ckpt):
            print(f"Error: No checkpoint found at {ckpt}")
            return
        system.load_checkpoint(ckpt)
        system.run_test(test_loader, visualize=args.visualize)

if __name__ == "__main__":
    main()