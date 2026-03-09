#!/usr/bin/env python3
# -- coding: utf-8 --

import rospy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import message_filters
from collections import deque
from sensor_msgs.msg import Image
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from scipy.interpolate import splprep, splev  # 用于轨迹平滑
from cv_bridge import CvBridge
from ultralytics import YOLO
import math
import os
import tf
import tf.transformations as tf_trans
import random
import colorsys
from scipy.interpolate import make_interp_spline # 用于曲线拟合

# =========================================================================
# 0. 全局配置 & 可视化参数
# =========================================================================

# --- [配置] 保存路径 ---
SAVE_DIR_ROOT = "/home/gaohaowen/Pose&Scene/data_collection/saved_results" # 请修改为你想要的路径

# --- [配置] 1. 墙壁位置配置 ---
WALL_CONFIG = {
    "X_MIN": -15.0, "X_MAX": 15.0, "Y_LEFT": 4.0, "Y_RIGHT": -3.6, 
    "HEIGHT_MIN": 0.8, "HEIGHT_MAX": 2.8 
}

# --- [配置] 2. 障碍物真实数据 ---
# OBSTACLES_DATA = [
# ]

# OBSTACLES_DATA = [
#     (1.2, 0.3, 0.8, 1.2)
# ]

OBSTACLES_DATA = [
    (0.88, -2.29, 0.4, 2.5), (4.02, -2.13, 0.4, 1.8), (7.0, -0.54, 0.35, 1.2),
    (5.74, 2.14, 0.5, 2.3), (4.02, 0.20, 0.5, 1.2), (2.66, 2.84, 0.4, 1.2),
    (-1.22, 2.49, 0.4, 1.2), (0.0, 0.8, 0.35, 1.2)
]
# OBSTACLES_DATA = [(0.34, 0.5, 0.5, 1.2)]

# --- 3. 视觉风格参数 ---
VOXEL_SIZE = 0.22
COLOR_HEIGHT_MAX = 2.5

# --- 4. 轨迹可视化参数 ---
VIS_DRONE_TRAJ_COLOR = (0.0, 0.0, 1.0, 1.0) # 蓝
VIS_DRONE_TRAJ_SCALE = 0.08                 
VIS_TRAJ_TIME  = 20.0                       

VIS_TARGET_TRAJ_COLOR = (1.0, 0.0, 0.0, 1.0) # 红
VIS_TARGET_TRAJ_SCALE = 0.08

VIS_PRED_POINT_COLOR = (0.6, 0.0, 0.8, 1.0) # 紫
VIS_PRED_POINT_SCALE = 0.04                 

VIS_VERIFY_LINE_COLOR = (0.0, 1.0, 0.0, 1.0) # 绿
VIS_VERIFY_LINE_SCALE = 0.01                 
VIS_VERIFY_LINE_DENSITY = 50                 

# --- 其他原有配置 ---
IMG_W, IMG_H = 640.0, 480.0
MAX_DIST_MM = 9000.0
DEPTH_ALIGN_U_OFFSET = 3.0
QUEUE_MAX_LEN = 100
HISTORY_STEPS = 5
SAMPLE_INTERVAL = 1.0 / 30.0 
INFERENCE_STRIDE = 2         
HISTORY_DT = 0.3             
VICON_Z_OFFSET = 0.75        
CONF_PRED_HORIZON = 5 

K_RGB = np.array([[385.52737, 0.0, 318.48364], [0.0, 385.12570, 249.94864], [0.0, 0.0, 1.0]])
K_DEPTH = np.array([[392.27487, 0.0, 321.56458], [0.0, 392.27487, 239.04189], [0.0, 0.0, 1.0]])

# =========================================================================
# 1. 网络模型定义 (保持不变)
# =========================================================================
class NeuralKalmanFusion(nn.Module):
    def __init__(self, motion_dim, obs_dim, hidden_dim):
        super(NeuralKalmanFusion, self).__init__()
        self.gate_net = nn.Sequential(nn.Linear(motion_dim + obs_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, hidden_dim), nn.Sigmoid())
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, h_motion, h_obs):
        m_emb = self.motion_proj(h_motion); o_emb = self.obs_proj(h_obs)
        concat_feat = torch.cat([h_motion, h_obs], dim=1)
        K = self.gate_net(concat_feat)
        return self.norm((1 - K) * m_emb + K * o_emb), K

class Model_NoSAOM_NoHeading(nn.Module):
    def __init__(self, pred_horizon=CONF_PRED_HORIZON, num_intentions=3, hidden_dim=270):
        super(Model_NoSAOM_NoHeading, self).__init__()
        self.pred_horizon = pred_horizon
        self.register_buffer('norm_step', torch.tensor(1.0)); self.register_buffer('norm_vel', torch.tensor(5.0)); self.register_buffer('norm_depth', torch.tensor(10.0))
        self.depth_cnn_backbone = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.visual_fc = nn.Sequential(nn.Conv2d(64, 16, 1), nn.Flatten(), nn.Linear(640, 128), nn.ReLU(inplace=True))
        self.vel_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.traj_embed = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.traj_gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.nkf_fusion = NeuralKalmanFusion(motion_dim=96, obs_dim=128, hidden_dim=hidden_dim)
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, pred_horizon * 3 * 2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, num_intentions))

    def normalize_data(self, state, traj_seq):
        norm_state = state.clone(); norm_state[:, :3] /= self.norm_vel; norm_state[:, 3] /= self.norm_depth
        norm_traj = traj_seq.clone(); norm_traj[:, :, :3] /= self.norm_step
        return norm_state, norm_traj

    def forward(self, depth, state, traj_seq):
        batch_size = depth.size(0)
        s_norm, t_norm = self.normalize_data(state, traj_seq)
        velocity = s_norm[:, :3]
        _, h_traj = self.traj_gru(self.traj_embed(t_norm))
        h_motion = torch.cat([h_traj[-1], self.vel_encoder(velocity)], dim=1)
        h_obs = self.visual_fc(self.depth_cnn_backbone(depth))
        fused_emb, _ = self.nkf_fusion(h_motion, h_obs)
        raw_params = self.regressor(fused_emb).view(batch_size, self.pred_horizon, 3, 2)
        return raw_params[..., 0], F.softplus(raw_params[..., 1]) + 0.01, self.classifier(fused_emb)

    def inference(self, depth, state, traj_seq):
        self.eval()
        with torch.no_grad():
            mu, sigma, intent_logits = self.forward(depth, state, traj_seq)
            return mu * self.norm_step, torch.argmax(intent_logits, dim=1)

# =========================================================================
# 2. 辅助工具类 (保持不变)
# =========================================================================
class DepthAligner:
    def __init__(self):
        self.fx_c = K_RGB[0, 0]; self.fy_c = K_RGB[1, 1]; self.cx_c = K_RGB[0, 2]; self.cy_c = K_RGB[1, 2]
        self.fx_d = K_DEPTH[0, 0]; self.fy_d = K_DEPTH[1, 1]; self.cx_d = K_DEPTH[0, 2]; self.cy_d = K_DEPTH[1, 2]
        self.baseline = 0.015; self.kernel_size = 5
    def get_robust_depth_val(self, u, v, depth_img):
        h, w = depth_img.shape; u_int, v_int = int(u), int(v)
        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h: return None
        r = self.kernel_size // 2
        u_min = max(0, u_int - r); u_max = min(w, u_int + r + 1)
        v_min = max(0, v_int - r); v_max = min(h, v_int + r + 1)
        patch = depth_img[v_min:v_max, u_min:u_max]
        valid = patch[patch > 0]
        return np.median(valid)/1000.0 if len(valid)>0 else None
    def align_point(self, u, v, depth_img):
        d = self.get_robust_depth_val(u, v, depth_img)
        if d is None or d < 0.1: return None
        x_n = (u-self.cx_c)/self.fx_c; x_nd = x_n + (self.baseline/d)
        return self.get_robust_depth_val(x_nd*self.fx_d+self.cx_d+DEPTH_ALIGN_U_OFFSET, (v-self.cy_c)/self.fy_c*self.fy_d+self.cy_d, depth_img) or d
    def get_aligned_depth_roi(self, depth_img, rgb_bbox):
        cx, cy, w, h = rgb_bbox
        d = self.align_point(cx, cy, depth_img) or 2.0
        x_n = (cx-self.cx_c)/self.fx_c; x_nd = x_n + (self.baseline/d)
        return [x_nd*self.fx_d+self.cx_d+DEPTH_ALIGN_U_OFFSET, (cy-self.cy_c)/self.fy_c*self.fy_d+self.cy_d, w*(self.fx_d/self.fx_c), h*(self.fy_d/self.fy_c)], d

class CoordinateTransformer:
    def __init__(self):
        self.T_body_cam = np.eye(4); R_yaw = tf_trans.euler_matrix(0, 0, math.radians(9.0))
        self.T_body_cam[0:3, 3] = [0.06, -0.015, -0.01]; self.T_body_cam[0:3, 0:3] = R_yaw[0:3, 0:3]
        self.R_opt_to_flu = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    def transform_point(self, point_3d, T): return np.dot(T, np.append(point_3d[:3], 1.0))[:3]

class VelocityEstimator:
    def __init__(self, window_time=0.6): self.window_time = window_time
    def estimate_velocity_at_time(self, current_time, history_queue):
        if len(history_queue) < 3: return np.zeros(3)
        pts = []; times = []
        for t, p_w in history_queue:
            if 0 <= current_time - t <= self.window_time: pts.append(p_w); times.append(t)
        if len(pts) < 3: return np.zeros(3)
        pts = np.array(pts); t_rel = np.array(times) - current_time
        try: return np.array([np.polyfit(t_rel, pts[:,i], 2)[1] for i in range(3)])
        except: return np.zeros(3)

class DataProcessor:
    def __init__(self): self.target_w, self.target_h = 160, 64; self.aligner = DepthAligner()
    def process_depth_roi(self, depth_img, rgb_bbox, person_depth_m):
        bbox_d, d_center = self.aligner.get_aligned_depth_roi(depth_img, rgb_bbox)
        cx, cy, w, h = bbox_d; roi_w, roi_h = w * 6.0, h * 1.2; img_h, img_w = depth_img.shape
        x1 = int(max(0, cx - roi_w/2)); y1 = int(max(0, cy - roi_h/2)); x2 = int(min(img_w, cx + roi_w/2)); y2 = int(min(img_h, cy + roi_h/2))
        if x2 <= x1 or y2 <= y1: return None
        depth_patch = depth_img[y1:y2, x1:x2].copy(); depth_patch[depth_patch > MAX_DIST_MM] = MAX_DIST_MM
        depth_patch[(depth_patch > 0) & (depth_patch < 300.0)] = 300.0
        patch_resized = cv2.resize(depth_patch, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        rel_depth = (patch_resized.astype(np.float32) / MAX_DIST_MM) - ((person_depth_m * 1000.0) / MAX_DIST_MM)
        rel_depth[patch_resized == 0] = 0.5
        mask = np.zeros((self.target_h, self.target_w), dtype=np.float32)
        scale_x = self.target_w / (x2 - x1); scale_y = self.target_h / (y2 - y1)
        bx1 = int((max(0, cx - w/2) - x1) * scale_x); by1 = int((max(0, cy - h/2) - y1) * scale_y)
        bx2 = int((min(img_w, cx + w/2) - x1) * scale_x); by2 = int((min(img_h, cy + h/2) - y1) * scale_y)
        mask[max(0, by1):min(self.target_h, by2), max(0, bx1):min(self.target_w, bx2)] = 1.0
        return np.stack([rel_depth, mask], axis=0)

# =========================================================================
# 3. 核心节点: 包含高保真体素可视化逻辑
# =========================================================================

class TrajectoryPredictorNode:
    def __init__(self):
        rospy.init_node('traj_predictor_node', anonymous=True)
        
        # 1. 模型加载
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model_NoSAOM_NoHeading().to(self.device)
        ckpt_path = "/home/gaohaowen/Pose&Scene/checkpoints_v1/no_SAOM_heading_1.5/best_model.pth"
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model_state_dict'])
            rospy.loginfo("Model Loaded")
        self.model.eval()

        # 2. 工具类
        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8n-pose.pt')
        self.processor = DataProcessor()
        self.transformer = CoordinateTransformer()
        self.vel_estimator = VelocityEstimator(window_time=0.6)
        
        # 3. 状态变量
        self.MAX_TIME_GAP = 0.5 
        self.world_pos_queue = deque(maxlen=QUEUE_MAX_LEN) 
        self.frame_counter = 0
        
        FPS = 30.0 
        
        # 1. 实际轨迹 (30Hz)
        self.drone_path_points = deque(maxlen=int(VIS_TRAJ_TIME * FPS))
        self.target_path_points = deque(maxlen=int(VIS_TRAJ_TIME * FPS))
        
        # 2. 预测散点 (10Hz)
        pred_freq = FPS / INFERENCE_STRIDE
        self.pred_points = deque(maxlen=int(VIS_TRAJ_TIME * pred_freq))
        
        # 3. 验证线 (低频)
        verify_freq = FPS / VIS_VERIFY_LINE_DENSITY
        verify_len = int(VIS_TRAJ_TIME * verify_freq * 2) + 10
        self.verify_lines_points = deque(maxlen=verify_len) 

        # 4. TF 矩阵
        T_body_to_cam_phys = np.eye(4)
        T_body_to_cam_phys[0:3, 3] = [0.06, -0.015, -0.01]   
        R_yaw = tf_trans.euler_matrix(0, 0, math.radians(9.0))
        T_body_to_cam_phys[0:3, 0:3] = R_yaw[0:3, 0:3]
        T_phys_to_opt = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        T_body_to_opt_mat = np.dot(T_phys_to_opt, np.linalg.inv(T_body_to_cam_phys))
        self.T_opt_to_body_mat = np.linalg.inv(T_body_to_opt_mat) 

        # 5. 可视化发布器
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.pub_drone_path = rospy.Publisher('/viz/drone_path', Marker, queue_size=1)
        self.pub_target_path = rospy.Publisher('/viz/target_path', Marker, queue_size=1)
        self.pub_pred_arrow = rospy.Publisher('/viz/pred_arrow', Marker, queue_size=1)

        self.pub_pred_points = rospy.Publisher('/viz/pred_points', Marker, queue_size=1)
        self.pub_verify_lines = rospy.Publisher('/viz/verify_lines', Marker, queue_size=1)
        self.pub_obstacles = rospy.Publisher('/viz/obstacles_voxel', Marker, queue_size=1, latch=True) 
        self.pub_path_msg = rospy.Publisher('/pred_traj_world', Path, queue_size=1)

        # 6. [NEW] 初始化保存目录
        self.rgb_save_dir = os.path.join(SAVE_DIR_ROOT, "rgb_yolo")
        self.depth_save_dir = os.path.join(SAVE_DIR_ROOT, "depth_pred")
        if not os.path.exists(self.rgb_save_dir): os.makedirs(self.rgb_save_dir)
        if not os.path.exists(self.depth_save_dir): os.makedirs(self.depth_save_dir)
        rospy.loginfo(f"Data will be saved to: {SAVE_DIR_ROOT}")

        # 订阅者
        rospy.Subscriber('/vicon/odom', Odometry, self.odom_tf_callback)
        sub_rgb = message_filters.Subscriber('/camera/color/image_raw', Image)
        sub_depth = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        sub_drone = message_filters.Subscriber('/vicon/odom', Odometry) 
        sub_target = message_filters.Subscriber('/vicon/odom/target', PoseStamped)
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth, sub_drone, sub_target], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback_4way)
        
        self.publish_voxel_obstacles()
        rospy.loginfo(f"🚀 Node Started. Traj Time: {VIS_TRAJ_TIME}s")

    # [NEW] 辅助函数：将世界坐标点投影到像素坐标
    def project_world_to_pixel(self, points_w, T_w_to_opt):
        pixels = []
        for p_w in points_w:
            # 1. World -> Optical Frame (Camera Frame)
            p_cam = np.dot(T_w_to_opt, np.append(p_w, 1.0))[:3]
            
            # 2. Camera -> Pixel (Using K_DEPTH)
            # z check to avoid division by zero or negative z
            if p_cam[2] <= 0.1: 
                continue
                
            fx, fy = K_DEPTH[0, 0], K_DEPTH[1, 1]
            cx, cy = K_DEPTH[0, 2], K_DEPTH[1, 2]
            
            u = int((p_cam[0] * fx / p_cam[2]) + cx)
            v = int((p_cam[1] * fy / p_cam[2]) + cy)
            
            pixels.append((u, v))
        return pixels

    def publish_voxel_obstacles(self):
        # ... (此处省略，保持原样，未修改) ...
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "voxels"
        marker.id = 0
        marker.type = Marker.CUBE_LIST  
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = VOXEL_SIZE * 0.92; marker.scale.y = VOXEL_SIZE * 0.92; marker.scale.z = VOXEL_SIZE * 0.92
        marker.color.a = 1.0 
        points = []; colors = []
        def get_color_by_height(z):
            ratio = min(max(z / COLOR_HEIGHT_MAX, 0.0), 1.0)
            hue = (1.0 - ratio) * 0.66 
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            return ColorRGBA(r, g, b, 1.0)
        for x in np.arange(WALL_CONFIG["X_MIN"], WALL_CONFIG["X_MAX"], VOXEL_SIZE):
            base_y = WALL_CONFIG["Y_LEFT"]
            thickness = random.randint(2, 4)
            for i in range(thickness):
                y = base_y + (i * VOXEL_SIZE) + random.uniform(-0.05, 0.05)
                base_h = (WALL_CONFIG["HEIGHT_MAX"] + WALL_CONFIG["HEIGHT_MIN"]) / 2
                h_noise = random.uniform(-0.4, 0.6) 
                final_h = max(base_h + h_noise, WALL_CONFIG["HEIGHT_MIN"]) 
                for z in np.arange(VOXEL_SIZE/2, final_h, VOXEL_SIZE):
                    points.append(Point(x, y, z)); colors.append(get_color_by_height(z))
            base_y_r = WALL_CONFIG["Y_RIGHT"]
            thickness_r = random.randint(2, 4)
            for i in range(thickness_r):
                y = base_y_r - (i * VOXEL_SIZE) + random.uniform(-0.05, 0.05)
                base_h = (WALL_CONFIG["HEIGHT_MAX"] + WALL_CONFIG["HEIGHT_MIN"]) / 2
                h_noise = random.uniform(-0.4, 0.6)
                final_h = max(base_h + h_noise, WALL_CONFIG["HEIGHT_MIN"])
                for z in np.arange(VOXEL_SIZE/2, final_h, VOXEL_SIZE):
                    points.append(Point(x, y, z)); colors.append(get_color_by_height(z))
        for obs in OBSTACLES_DATA:
            ox, oy, radius, height_cfg = obs
            grid_range = np.arange(-radius - 0.2, radius + 0.2 + 0.01, VOXEL_SIZE)
            for dx in grid_range:
                for dy in grid_range:
                    dist_sq = dx*dx + dy*dy
                    max_r_sq = radius * radius
                    if dist_sq <= max_r_sq:
                        if dist_sq > max_r_sq * 0.8 and random.random() > 0.8: continue
                        vx = ox + dx; vy = oy + dy
                        local_h = max(height_cfg + random.uniform(-0.4, 0.2), VOXEL_SIZE)
                        for z in np.arange(VOXEL_SIZE/2, local_h, VOXEL_SIZE):
                            points.append(Point(vx, vy, z)); colors.append(get_color_by_height(z))
        marker.points = points; marker.colors = colors
        self.pub_obstacles.publish(marker)

    def pose_msg_to_matrix(self, msg):
        if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'): p = msg.pose.pose.position; q = msg.pose.pose.orientation
        elif hasattr(msg, 'pose'): p = msg.pose.position; q = msg.pose.orientation
        else: return np.eye(4)
        T = tf_trans.quaternion_matrix([q.x, q.y, q.z, q.w]); T[0:3, 3] = [p.x, p.y, p.z]
        return T

    def transform_point(self, point_3d, T): return np.dot(T, np.append(point_3d[:3], 1.0))[:3]
    
    def odom_tf_callback(self, drone_msg):
        pos = (drone_msg.pose.pose.position.x, drone_msg.pose.pose.position.y, drone_msg.pose.pose.position.z)
        ori = (drone_msg.pose.pose.orientation.x, drone_msg.pose.pose.orientation.y, drone_msg.pose.pose.orientation.z, drone_msg.pose.pose.orientation.w)
        self.tf_broadcaster.sendTransform(pos, ori, rospy.Time.now(), "base_link", "map")

    def publish_points_list(self, publisher, points, color_rgba, scale, ns, timestamp, duration):
        if not points: return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns; marker.id = 0; marker.type = Marker.SPHERE_LIST; marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale; marker.scale.y = scale; marker.scale.z = scale
        marker.color = ColorRGBA(*color_rgba)
        marker.lifetime = rospy.Duration(duration)
        msg_points = [Point(p[0], p[1], p[2]) for p in points]
        marker.points = msg_points
        publisher.publish(marker)

    def publish_line_strip(self, publisher, points, color_rgba, scale, ns, timestamp, duration):
        if not points: return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns; marker.id = 0; marker.type = Marker.LINE_STRIP; marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.color = ColorRGBA(*color_rgba)
        marker.lifetime = rospy.Duration(duration)
        msg_points = [Point(p[0], p[1], p[2]) for p in points]
        marker.points = msg_points
        publisher.publish(marker)

    def publish_arrow_custom(self, publisher, position, orientation, color_rgba, scale_vec, ns):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns; marker.id = 0; marker.type = Marker.ARROW; marker.action = Marker.ADD
        marker.pose.position.x = position[0]; marker.pose.position.y = position[1]
        marker.pose.position.z = position[2] + 0.01 
        if isinstance(orientation, Quaternion): marker.pose.orientation = orientation
        else: marker.pose.orientation = Quaternion(*orientation)
        marker.scale.x = scale_vec.x; marker.scale.y = scale_vec.y; marker.scale.z = scale_vec.z
        marker.color = ColorRGBA(*color_rgba)
        marker.lifetime = rospy.Duration(0.1)
        publisher.publish(marker)

    def sync_callback_4way(self, rgb_msg, depth_msg, drone_msg, target_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e: 
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        current_time = rgb_msg.header.stamp.to_sec()
        timestamp_str = "{:.6f}".format(current_time) # 用于保存文件名

        curr_drone_mat = self.pose_msg_to_matrix(drone_msg)
        curr_target_mat = self.pose_msg_to_matrix(target_msg)
        vicon_pos_w = curr_target_mat[0:3, 3]; vicon_pos_w[2] -= VICON_Z_OFFSET 
        drone_pos_w = curr_drone_mat[0:3, 3]

        # 1. 基础轨迹绘制
        self.drone_path_points.append(drone_pos_w)
        self.target_path_points.append(vicon_pos_w)
        self.publish_line_strip(self.pub_drone_path, self.drone_path_points, VIS_DRONE_TRAJ_COLOR, VIS_DRONE_TRAJ_SCALE, "drone_trail", current_time, VIS_TRAJ_TIME)
        self.publish_line_strip(self.pub_target_path, self.target_path_points, VIS_TARGET_TRAJ_COLOR, VIS_TARGET_TRAJ_SCALE, "target_trail", current_time, VIS_TRAJ_TIME)

        # 2. 维护队列
        if not self.world_pos_queue: 
            self.world_pos_queue.append((current_time, vicon_pos_w))
        else:
            last_t = self.world_pos_queue[-1][0]
            if current_time < (last_t - 1.0): 
                rospy.logwarn("Loop detected! Resetting buffers.")
                self.world_pos_queue.clear()
                self.verify_lines_points.clear()
                self.world_pos_queue.append((current_time, vicon_pos_w)) 
                return 
            if current_time <= last_t: return
            self.world_pos_queue.append((current_time, vicon_pos_w))

        # 3. 验证线 (实时连接)
        if self.verify_lines_points:
            m = Marker()
            m.header.frame_id="map"; m.header.stamp=rospy.Time.now(); m.ns="verify_lines"; m.id=0; m.type=Marker.LINE_LIST; m.action=Marker.ADD
            m.scale.x=VIS_VERIFY_LINE_SCALE; m.color=ColorRGBA(*VIS_VERIFY_LINE_COLOR)
            m.lifetime = rospy.Duration(0.1)
            m.points=list(self.verify_lines_points) 
            self.pub_verify_lines.publish(m)

        # 4. 推理前置检查 & YOLO
        results = self.yolo(cv_rgb, verbose=False, classes=[0])
        if len(results) == 0 or len(results[0].boxes) == 0: return 
        r = results[0]; box = r.boxes.xywh.cpu().numpy()[0]
        
        # [NEW: Feature 1] 保存带有 YOLO 绘制的 RGB 图像
        # plot() 返回 BGR numpy 数组
        rgb_with_pose = r.plot() 
        
        if len(self.world_pos_queue) < 2: return
        time_span = self.world_pos_queue[-1][0] - self.world_pos_queue[0][0]
        if time_span < (HISTORY_STEPS * HISTORY_DT): return

        self.frame_counter += 1
        if self.frame_counter % INFERENCE_STRIDE != 1: return

        T_opt_to_world = np.dot(curr_drone_mat, self.T_opt_to_body_mat)
        T_world_to_opt_curr = np.linalg.inv(T_opt_to_world) # 用于将世界坐标转回相机坐标
        vel_w = self.vel_estimator.estimate_velocity_at_time(current_time, self.world_pos_queue)
        vel_cam = np.dot(T_world_to_opt_curr[0:3, 0:3], vel_w)

        points_seq_cam = [self.transform_point(vicon_pos_w, T_world_to_opt_curr)]
        valid_hist = True
        for h in range(1, HISTORY_STEPS + 1):
            target_t = current_time - h * HISTORY_DT
            closest = min(self.world_pos_queue, key=lambda x: abs(x[0] - target_t))
            if abs(closest[0] - target_t) > 0.5: valid_hist = False; break
            points_seq_cam.append(self.transform_point(closest[1], T_world_to_opt_curr))
        
        if not valid_hist: return
        points_seq_cam.reverse()
        traj_deltas = np.diff(np.array(points_seq_cam), axis=0)

        u, v = box[0], box[1]
        curr_person_depth_m = self.processor.aligner.align_point(u, v, cv_depth) or 2.0
        depth_tensor = self.processor.process_depth_roi(cv_depth, box, curr_person_depth_m)
        if depth_tensor is None: return

        # 5. 执行推理
        state_vec = np.concatenate([vel_cam, [curr_person_depth_m]])
        t_in = [
            torch.from_numpy(depth_tensor).unsqueeze(0).float().to(self.device),
            torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device),
            torch.cat([torch.from_numpy(traj_deltas).unsqueeze(0).float().to(self.device), 
                       torch.full((1, 5, 1), HISTORY_DT, device=self.device)], dim=-1)
        ]
        with torch.no_grad(): pred_deltas_m, _ = self.model.inference(*t_in); pred_deltas_m = pred_deltas_m.cpu().numpy()[0]

        path_msg = Path(); path_msg.header = rgb_msg.header; path_msg.header.frame_id = "map"
        accumulated_pos_w = vicon_pos_w.copy()
        R_cam_to_w = T_opt_to_world[0:3, 0:3]
        pred_pos_15 = None 
        last_delta_w = None

        # [NEW] 收集所有预测的世界坐标点 (用于重投影)
        pred_points_w_list = [] 

        for i, delta_cam in enumerate(pred_deltas_m):
            delta_w = np.dot(R_cam_to_w, delta_cam)
            accumulated_pos_w += delta_w
            last_delta_w = delta_w
            
            pred_points_w_list.append(accumulated_pos_w.copy())

            pose = PoseStamped(); pose.header = path_msg.header
            pose.pose.position.x = accumulated_pos_w[0]
            pose.pose.position.y = accumulated_pos_w[1]
            pose.pose.position.z = accumulated_pos_w[2]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

            if i == 4: # 1.5s
                pred_pos_15 = accumulated_pos_w.copy()

        self.pub_path_msg.publish(path_msg)

        if len(pred_points_w_list) > 1:
            # A. 深度图可视化 (建议试试 cv2.COLORMAP_PLASMA 或 MAGMA，比 JET 更好看，不过这里保留 JET)
            depth_vis = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
            
            # B. 投影点到像素坐标
            pixels = self.project_world_to_pixel(pred_points_w_list, T_world_to_opt_curr)
            
            if len(pixels) >= 2:
                pts = np.array(pixels, np.int32)
                
                # C. 轨迹平滑 (B-Spline Interpolation)
                smooth_pts = pts
                try:
                    # 只有点数 >= 4 时才能进行优秀的三次样条插值
                    if len(pts) >= 4:
                        # 去除相邻重复点，防止插值算法崩溃
                        _, idx = np.unique(pts, axis=0, return_index=True)
                        pts_unique = pts[np.sort(idx)]
                        
                        if len(pts_unique) >= 4:
                            x, y = pts_unique[:, 0], pts_unique[:, 1]
                            tck, u = splprep([x, y], s=0.0, k=3) # k=3 表示三次样条
                            u_new = np.linspace(0, 1, 100)       # 密集插值出 100 个点
                            x_new, y_new = splev(u_new, tck)
                            smooth_pts = np.vstack((x_new, y_new)).T.astype(np.int32)
                except Exception as e:
                    # 如果插值失败(例如所有点共线)，退回原始折线
                    smooth_pts = pts
                
                smooth_pts = smooth_pts.reshape((-1, 1, 2))
                pts_raw = pts.reshape((-1, 1, 2))

                # D. 绘制描边轨迹 (双层绘制法提升对比度)
                # 1. 粗黑色底层 (充当阴影/描边)，极大增强背景隔离度
                cv2.polylines(depth_vis, [smooth_pts], isClosed=False, color=(0, 0, 0), thickness=6, lineType=cv2.LINE_AA)
                # 2. 细亮色表层 (明黄色)
                cv2.polylines(depth_vis, [smooth_pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                
                # E. 绘制预测关键点 (散点 Node 效果)
                # 画出真实的预测点，体现这是"离散预测"而不是盲目的曲线
                for pt in pixels:
                    # 白底黑边的小圆点
                    cv2.circle(depth_vis, tuple(pt), radius=4, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(depth_vis, tuple(pt), radius=3, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

                if len(smooth_pts) >= 10:
                    # 取平滑曲线上倒数第10个点作为箭头起点，让箭头方向更准
                    # 强转为 int，防止 OpenCV 偶尔对 numpy 标量类型报错
                    arrow_start = (int(smooth_pts[-10][0][0]), int(smooth_pts[-10][0][1])) 
                else:
                    arrow_start = (int(pixels[-2][0]), int(pixels[-2][1]))
                    
                arrow_end = (int(pixels[-1][0]), int(pixels[-1][1]))
                
                if arrow_start != arrow_end:
                    # 箭头同样采用双层绘制 (注意参数名是 line_type)
                    cv2.arrowedLine(depth_vis, arrow_start, arrow_end, color=(0, 0, 0), thickness=6, tipLength=0.4, line_type=cv2.LINE_AA)
                    cv2.arrowedLine(depth_vis, arrow_start, arrow_end, color=(0, 0, 255), thickness=2, tipLength=0.4, line_type=cv2.LINE_AA)

            # E. 保存图片
            cv2.imwrite(os.path.join(self.rgb_save_dir, f"{timestamp_str}.jpg"), rgb_with_pose)
            cv2.imwrite(os.path.join(self.depth_save_dir, f"{timestamp_str}.jpg"), depth_vis)

        # 箭头 (RViz)
        if last_delta_w is not None:
            yaw = math.atan2(last_delta_w[1], last_delta_w[0])
            q = tf_trans.quaternion_from_euler(0, 0, yaw)
            arrow_scale = Vector3(0.8, 0.10, 0.001) 
            self.publish_arrow_custom(self.pub_pred_arrow, accumulated_pos_w, q, VIS_PRED_POINT_COLOR, arrow_scale, "pred_end_arrow")

        # 散点 (1.5s) 和 验证线
        if pred_pos_15 is not None:
            self.pred_points.append(pred_pos_15)
            self.publish_points_list(self.pub_pred_points, self.pred_points, VIS_PRED_POINT_COLOR, VIS_PRED_POINT_SCALE, "pred_points", current_time, VIS_TRAJ_TIME)
            
            if self.frame_counter % VIS_VERIFY_LINE_DENSITY == 1:
                self.verify_lines_points.append(Point(*drone_pos_w))
                self.verify_lines_points.append(Point(*pred_pos_15))

if __name__ == '__main__':
    try: TrajectoryPredictorNode(); rospy.spin()
    except rospy.ROSInterruptException: pass