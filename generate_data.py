#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import cv2
import os
import json
import math
import tf.transformations as tf_trans
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import deque

# [NEW] 引入 matplotlib 库用于绘制轨迹图 (强制使用Agg后端，防止服务器报错)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 批量任务配置 =================
BAG_TASKS =[
    ("/home/gaohaowen/scripts/bagfiles/data_collect_01.bag", "./Data_1.5s/dataset_processed_01"),
    ("/home/gaohaowen/scripts/bagfiles/data_collect_02.bag", "./Data_1.5s/dataset_processed_02"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_03.bag", "./Data_1.5s/dataset_processed_03"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_04.bag", "./Data_1.5s/dataset_processed_04"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_05.bag", "./Data_1.5s/dataset_processed_05"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_06.bag", "./Data_1.5s/dataset_processed_06"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_07.bag", "./Data_1.5s/dataset_processed_07"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_08.bag", "./Data_1.5s/dataset_processed_08"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_09.bag", "./Data_1.5s/dataset_processed_09"),
    # ("/home/gaohaowen/scripts/bagfiles/data_collect_10.bag", "./Data_1.5s/dataset_processed_10"),
]

# ================= 参数配置 =================
HUMAN_TOPIC = "/vicon/odom/target"      
DRONE_TOPIC = "/vicon/odom"             
RGB_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_rect_raw"

K_RGB = np.array([[385.52737, 0.0,       318.48364],[0.0,       385.12570, 249.94864],[0.0,       0.0,       1.0]
])

K_DEPTH = np.array([[392.27487, 0.0,       321.56458],[0.0,       392.27487, 239.04189],[0.0,       0.0,       1.0]
])

# [CONFIG] 手动水平对齐补偿 (Global Offset)
DEPTH_ALIGN_U_OFFSET = 3.0  

# [NEW CONFIG] 肩膀内缩比例 (避免边缘噪声)
SHOULDER_INWARD_RATIO = 0.18

# [NEW CONFIG] 肩膀下移比例 (垂直方向避免头部/背景)
# 例如 0.05 表示向下移动肩宽的 5%
SHOULDER_DOWN_RATIO = 0.4


T_body_to_cam_phys = np.eye(4)
T_body_to_cam_phys[0:3, 3] = [0.06, -0.015, -0.01]   
R_yaw = tf_trans.euler_matrix(0, 0, math.radians(9.0))
T_body_to_cam_phys[0:3, 0:3] = R_yaw[0:3, 0:3]

T_phys_to_opt = np.array([[0, -1, 0, 0],[0, 0, -1, 0], [1, 0, 0, 0],[0, 0, 0, 1]])
T_body_to_opt_mat = np.dot(T_phys_to_opt, np.linalg.inv(T_body_to_cam_phys))
T_opt_to_body_mat = np.linalg.inv(T_body_to_opt_mat) 

BAG_FPS = 30.0
HISTORY_DT = 0.3; HISTORY_STRIDE = int(HISTORY_DT * BAG_FPS); HISTORY_STEPS = 5
FUTURE_DT = 0.3; FUTURE_STRIDE = int(FUTURE_DT * BAG_FPS); FUTURE_STEPS = 5
DATA_SAMPLE_STRIDE = 1
VICON_Z_OFFSET = 0.75            
VALIDATION_PIXEL_THRESH = 60.0  
DEBUG_SAVE_STRIDE = 10   
DEBUG_MAX_IMAGES = 500          

# ================= 工具函数 =================
def pose_msg_to_matrix(msg):
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
    elif hasattr(msg, 'pose'):
        p = msg.pose.position
        q = msg.pose.orientation
    else:
        return np.eye(4)
    T = tf_trans.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[0:3, 3] = [p.x, p.y, p.z]
    return T

def transform_point(point_3d, T):
    p = np.append(point_3d[:3], 1.0)
    return np.dot(T, p)[:3]

def project_world_to_pixel(point_world, drone_pose_matrix):
    try:
        T_body_world = np.linalg.inv(drone_pose_matrix)
    except:
        return None
    p_w_homo = np.append(point_world, 1.0)
    p_body = np.dot(T_body_world, p_w_homo) 
    p_opt = np.dot(T_body_to_opt_mat, p_body)
    x, y, z = p_opt[0], p_opt[1], p_opt[2]
    if z <= 0.1: return None 
    u = (x * K_RGB[0, 0] / z) + K_RGB[0, 2]
    v = (y * K_RGB[1, 1] / z) + K_RGB[1, 2]
    return np.array([u, v])

class VelocityEstimator:
    def __init__(self, window_time=0.6):
        self.window_time = window_time
        self.history = deque(maxlen=50) 
    def update(self, t, pos_world):
        self.history.append((t, pos_world))
    def get_velocity_in_current_camera_frame(self, current_time, T_world_to_opt_curr):
        valid_points = []
        times =[]
        for t, p_w in self.history:
            if current_time - t <= self.window_time and t <= current_time:
                valid_points.append(p_w)
                times.append(t)
        if len(valid_points) < 3 or (times[-1] - times[0]) < 0.1:
            return np.array([0.0, 0.0, 0.0])
        pts_w_homo = np.column_stack((np.array(valid_points), np.ones(len(valid_points))))
        pts_opt_homo = np.dot(T_world_to_opt_curr, pts_w_homo.T).T
        pts_opt = pts_opt_homo[:, :3] 
        t_rel = np.array(times) - current_time
        vx, vy, vz = 0.0, 0.0, 0.0
        try:
            coeff_x = np.polyfit(t_rel, pts_opt[:, 0], deg=2)
            coeff_y = np.polyfit(t_rel, pts_opt[:, 1], deg=2)
            coeff_z = np.polyfit(t_rel, pts_opt[:, 2], deg=2)
            vx = coeff_x[1]; vy = coeff_y[1]; vz = coeff_z[1]
        except np.linalg.LinAlgError:
            return np.array([0.0, 0.0, 0.0])
        return np.array([vx, vy, vz], dtype=np.float32)

class DepthAligner:
    def __init__(self):
        self.fx_c = K_RGB[0, 0]; self.fy_c = K_RGB[1, 1]
        self.cx_c = K_RGB[0, 2]; self.cy_c = K_RGB[1, 2]
        self.fx_d = K_DEPTH[0, 0]; self.fy_d = K_DEPTH[1, 1]
        self.cx_d = K_DEPTH[0, 2]; self.cy_d = K_DEPTH[1, 2]
        self.baseline = 0.015 
        self.kernel_size = 5

    def get_robust_depth_val(self, u, v, depth_img):
        h, w = depth_img.shape
        u_int, v_int = int(u), int(v)
        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h: return None
        r = self.kernel_size // 2
        u_min = max(0, u_int - r); u_max = min(w, u_int + r + 1)
        v_min = max(0, v_int - r); v_max = min(h, v_int + r + 1)
        patch = depth_img[v_min:v_max, u_min:u_max]
        valid_pixels = patch[patch > 0]
        if len(valid_pixels) == 0: return None
        return np.median(valid_pixels) / 1000.0

    def align_point(self, u, v, depth_img):
        d_val = self.get_robust_depth_val(u, v, depth_img)
        if d_val is None or d_val < 0.1: return None
        x_norm = (u - self.cx_c) / self.fx_c
        x_norm_d = x_norm + (self.baseline / d_val)
        u_d = x_norm_d * self.fx_d + self.cx_d + DEPTH_ALIGN_U_OFFSET
        v_d = (v - self.cy_c) / self.fy_c * self.fy_d + self.cy_d
        final_d = self.get_robust_depth_val(u_d, v_d, depth_img)
        if final_d is not None: return final_d
        return d_val

    def get_aligned_uv(self, u, v, depth_img):
        d_val = self.get_robust_depth_val(u, v, depth_img)
        if d_val is None or d_val < 0.1: return None, None, None
        x_norm = (u - self.cx_c) / self.fx_c
        x_norm_d = x_norm + (self.baseline / d_val)
        u_d = x_norm_d * self.fx_d + self.cx_d + DEPTH_ALIGN_U_OFFSET
        v_d = (v - self.cy_c) / self.fy_c * self.fy_d + self.cy_d
        final_d = self.get_robust_depth_val(u_d, v_d, depth_img)
        if final_d is None: final_d = d_val 
        return u_d, v_d, final_d

    def get_aligned_depth_roi(self, depth_img, rgb_bbox):
        cx, cy, w, h = rgb_bbox
        d_center = self.align_point(cx, cy, depth_img)
        if d_center is None: d_center = 2.0
        x_norm = (cx - self.cx_c) / self.fx_c
        x_norm_d = x_norm + (self.baseline / d_center)
        cx_d = x_norm_d * self.fx_d + self.cx_d + DEPTH_ALIGN_U_OFFSET
        cy_d = (cy - self.cy_c) / self.fy_c * self.fy_d + self.cy_d
        w_d = w * (self.fx_d / self.fx_c)
        h_d = h * (self.fy_d / self.fy_c)
        return[cx_d, cy_d, w_d, h_d]

class DataProcessor:
    def __init__(self):
        self.target_w = 160; self.target_h = 64
        self.MAX_DIST_MM = 9000.0; self.MIN_DIST_MM = 300.0   
        self.aligner = DepthAligner()
        self.vel_estimator = VelocityEstimator() 

    def get_centroid_and_heading(self, kpts, depth_img):
        Z_DIFF_GAIN = 1.5
        Z_DIFF_OFFSET = 0.00
        SIDEWAYS_SCORE_THRESH = 18.0  
        SIDEWAYS_OFFSET_GAIN = 0.06   

        modified_kpts_uv = {} 
        sideways_flag = 0 

        joint_pairs =[(5, 6, "shoulder"), (11, 12, "hip")]
        for idx1, idx2, name in joint_pairs:
            if kpts[idx1][2] > 0.5 and kpts[idx2][2] > 0.5:
                u1, v1 = kpts[idx1][:2]; u2, v2 = kpts[idx2][:2]
                vec_u = u2 - u1; vec_v = v2 - v1
                width = math.sqrt(vec_u**2 + vec_v**2)
                
                inward_ratio = SHOULDER_INWARD_RATIO
                down_ratio = SHOULDER_DOWN_RATIO
                
                u1_in = u1 + vec_u * inward_ratio; v1_in = v1 + vec_v * inward_ratio
                u2_in = u2 - vec_u * inward_ratio; v2_in = v2 - vec_v * inward_ratio
                down_offset = width * down_ratio
                v1_final = v1_in + down_offset; v2_final = v2_in + down_offset
                modified_kpts_uv[idx1] = (u1_in, v1_final)
                modified_kpts_uv[idx2] = (u2_in, v2_final)

        indices = [5, 6, 11, 12]
        points_3d = {}
        for idx in indices:
            if idx in modified_kpts_uv: u, v = modified_kpts_uv[idx]
            else: u, v = kpts[idx][:2]
            conf = kpts[idx][2]
            if conf < 0.5: continue
            z = self.aligner.align_point(u, v, depth_img)
            if z is None or z < 0.3 or z > 8.0: continue
            x = (u - K_RGB[0,2]) * z / K_RGB[0,0]
            y = (v - K_RGB[1,2]) * z / K_RGB[1,1]
            points_3d[idx] = np.array([x, y, z])

        if len(points_3d) < 3: return None, None, 0
        centroid = np.mean(list(points_3d.values()), axis=0)

        if 5 not in points_3d or 6 not in points_3d:
            return centroid, None, 0

        u_ls, _ = modified_kpts_uv.get(5, kpts[5][:2])
        u_rs, _ = modified_kpts_uv.get(6, kpts[6][:2])
        
        pixel_u_diff = u_rs - u_ls 
        person_depth = centroid[2] 
        sideways_score = abs(pixel_u_diff) * person_depth
        
        if sideways_score < SIDEWAYS_SCORE_THRESH:
            z_ls = points_3d[5][2]
            z_rs = points_3d[6][2]
            
            if z_rs < z_ls: base_heading_x = 1.0
            else:           base_heading_x = -1.0
            
            z_offset = pixel_u_diff * SIDEWAYS_OFFSET_GAIN * base_heading_x
            
            heading_sideways = np.array([base_heading_x, 0.0, z_offset])
            norm = np.linalg.norm(heading_sideways)
            if norm > 1e-3:
                heading_sideways /= norm
                sideways_flag = 1 
                return centroid, heading_sideways, sideways_flag

        p_ls = points_3d[5]
        p_rs = points_3d[6]

        raw_dx = p_rs[0] - p_ls[0]
        raw_dz = p_rs[2] - p_ls[2]

        if abs(raw_dz) > 1e-3: 
            sign_z = np.sign(raw_dz)
            amplified_dz = (raw_dz * Z_DIFF_GAIN) + (sign_z * Z_DIFF_OFFSET)
        else:
            amplified_dz = raw_dz

        guide_vec = np.array([-amplified_dz, 0.0, raw_dx])
        
        norm_guide = np.linalg.norm(guide_vec)
        if norm_guide < 1e-3: return centroid, None, 0
        guide_vec /= norm_guide 

        final_heading = None

        if 11 in points_3d and 12 in points_3d:
            p_lh = points_3d[11]; p_rh = points_3d[12]
            p_hip_center = (p_lh + p_rh) / 2.0
            p_shoulder_center = (p_ls + p_rs) / 2.0
            
            vec_a_modified = p_rs - p_ls
            vec_a_modified[2] = amplified_dz 
            
            vec_b = p_hip_center - p_shoulder_center 
            
            plane_normal = np.cross(vec_b, vec_a_modified)
            plane_normal[1] = 0.0 
            norm_plane = np.linalg.norm(plane_normal)
            
            if norm_plane > 0.1: 
                plane_normal /= norm_plane 
                if np.dot(plane_normal, guide_vec) < 0: final_heading = -plane_normal
                else:                                   final_heading = plane_normal

        if final_heading is None:
            final_heading = guide_vec
        
        final_heading[1] = 0.0
        final_heading /= np.linalg.norm(final_heading)

        return centroid, final_heading, sideways_flag

    def process_depth_roi(self, depth_img, bbox_d, person_depth_m):
        cx, cy, w, h = bbox_d
        img_h, img_w = depth_img.shape
        roi_w, roi_h = w * 6.0, h * 1.2 
        cy_offset = cy 
        x1, y1 = int(max(0, cx - roi_w/2)), int(max(0, cy_offset - roi_h/2))
        x2, y2 = int(min(img_w, cx + roi_w/2)), int(min(img_h, cy_offset + roi_h/2))
        if x2 <= x1 or y2 <= y1: return None, None

        depth_patch = depth_img[y1:y2, x1:x2].copy()
        depth_patch[depth_patch > self.MAX_DIST_MM] = self.MAX_DIST_MM
        depth_patch[(depth_patch > 0) & (depth_patch < self.MIN_DIST_MM)] = self.MIN_DIST_MM
        patch_resized = cv2.resize(depth_patch, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        depth_norm = patch_resized.astype(np.float32) / self.MAX_DIST_MM
        person_mm = max(self.MIN_DIST_MM, min(self.MAX_DIST_MM, person_depth_m * 1000.0))
        person_norm = person_mm / self.MAX_DIST_MM
        rel_depth = depth_norm - person_norm
        rel_depth[patch_resized == 0] = 0.5 

        mask = np.zeros((self.target_h, self.target_w), dtype=np.float32)
        scale_x = self.target_w / (x2 - x1)
        scale_y = self.target_h / (y2 - y1)
        bx1 = int((max(0, cx - w/2) - x1) * scale_x)
        by1 = int((max(0, cy - h/2) - y1) * scale_y)
        bx2 = int((min(img_w, cx + w/2) - x1) * scale_x)
        by2 = int((min(img_h, cy + h/2) - y1) * scale_y)
        mask[max(0, by1):min(self.target_h, by2), max(0, bx1):min(self.target_w, bx2)] = 1.0
        
        return np.stack([rel_depth, mask], axis=0), {'x1': x1, 'y1': y1, 'scale_x': scale_x, 'scale_y': scale_y}

class Visualizer:
    def __init__(self, output_dir):
        self.save_dir = os.path.join(output_dir, "debug_vis")
        os.makedirs(self.save_dir, exist_ok=True)
        self.saved_count = 0
        self.saved_traj_count = 0  # [NEW]

    # [MODIFIED] 修改绘制逻辑：分别保存RGB和Depth
    def draw_and_save(self, frame_idx, cv_rgb, bbox, kpts, proj_gt, hip_center, depth_feat, shoulder_pts_data, hip_vis_data=None, heading_vec=None, sideways_flag=0):
        if self.saved_count >= DEBUG_MAX_IMAGES: return

        # ==========================================
        # 1. 保存 RGB (带框和GT标志)
        # ==========================================
        vis_img = cv_rgb.copy()
        cx, cy, w, h = bbox
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if hip_center is not None:
            hc_x, hc_y = int(hip_center[0]), int(hip_center[1])
            cv2.drawMarker(vis_img, (hc_x, hc_y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        if proj_gt is not None:
            u_gt, v_gt = int(proj_gt[0]), int(proj_gt[1])
            cv2.circle(vis_img, (u_gt, v_gt), 6, (0, 255, 0), -1) 
            cv2.putText(vis_img, "GT", (u_gt+15, v_gt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(f"{self.save_dir}/{frame_idx:06d}_rgb.jpg", vis_img)

        # ==========================================
        # 2. 保存 Depth (带原先的所有肩部、heading标记)
        # ==========================================
        if depth_feat is not None:
            raw_depth = depth_feat[0] 
            d_norm = np.clip((raw_depth + 1.0) * 0.5 * 255, 0, 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

            target_w = 320; vis_scale = target_w / 160.0; target_h = int(64 * vis_scale) 
            
            d_vis_big = cv2.resize(d_color, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            pt_map = {} 
            for pt_data in shoulder_pts_data:
                px = int(pt_data[0] * vis_scale)
                py = int(pt_data[1] * vis_scale)
                label = pt_data[3]
                pt_map[label] = (px, py)
            
            hip_pt = None
            if hip_vis_data is not None:
                h_px = int(hip_vis_data[0] * vis_scale)
                h_py = int(hip_vis_data[1] * vis_scale)
                h_d = hip_vis_data[2]
                hip_pt = (h_px, h_py)
                if 0 <= h_px < target_w and 0 <= h_py < target_h:
                    cv2.circle(d_vis_big, (h_px, h_py), 4, (0, 255, 0), -1) 
                    cv2.circle(d_vis_big, (h_px, h_py), 5, (255, 255, 255), 1)
                    cv2.putText(d_vis_big, f"H:{h_d:.2f}m", (h_px - 20, h_py + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if "L" in pt_map and "R" in pt_map:
                cv2.line(d_vis_big, pt_map["L"], pt_map["R"], (255, 0, 255), 2)
                if hip_pt is not None:
                    cv2.line(d_vis_big, pt_map["L"], hip_pt, (255, 255, 255), 1)
                    cv2.line(d_vis_big, pt_map["R"], hip_pt, (255, 255, 255), 1)

            for pt_data in shoulder_pts_data:
                px = int(pt_data[0] * vis_scale)
                py = int(pt_data[1] * vis_scale)
                d_val = pt_data[2]
                label = pt_data[3]

                if 0 <= px < target_w and 0 <= py < target_h:
                    if label == "L": pt_color = (255, 0, 0)   
                    else:            pt_color = (255, 0, 255) 

                    cv2.circle(d_vis_big, (px, py), 4, pt_color, -1) 
                    cv2.circle(d_vis_big, (px, py), 5, (255, 255, 255), 1) 
                    
                    text = f"{label}:{d_val:.2f}m"
                    font = cv2.FONT_HERSHEY_SIMPLEX; scale = 0.4; thickness = 1
                    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                    
                    if label == "L": tx = px - text_w - 20; ty = py + text_h // 2
                    else:            tx = px + 20; ty = py + text_h // 2
                    
                    cv2.putText(d_vis_big, text, (tx, ty), font, scale, (0, 0, 0), thickness+1) 
                    cv2.putText(d_vis_big, text, (tx, ty), font, scale, (255, 255, 255), thickness)

            if heading_vec is not None:
                h_text = f"Heading X-Z: [{heading_vec[0]:.2f}, {heading_vec[2]:.2f}]"
                cv2.putText(d_vis_big, h_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(d_vis_big, h_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if sideways_flag == 1:
                    mode_text = "[SIDEWAYS MODE]"
                    text_y = target_h - 10
                    cv2.putText(d_vis_big, mode_text, (5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                start_x, start_y = target_w // 2, target_h // 2
                if "L" in pt_map and "R" in pt_map and hip_pt is not None:
                    start_x = int((pt_map["L"][0] + pt_map["R"][0] + hip_pt[0]) / 3)
                    start_y = int((pt_map["L"][1] + pt_map["R"][1] + hip_pt[1]) / 3)
                
                arrow_len = 40 
                dx = int(heading_vec[0] * arrow_len)
                end_pt = (start_x + dx, start_y) 
                arrow_color = (0, 0, 255) if sideways_flag == 1 else (0, 255, 255)
                
                cv2.arrowedLine(d_vis_big, (start_x, start_y), end_pt, arrow_color, 3, tipLength=0.3)

            cv2.imwrite(f"{self.save_dir}/{frame_idx:06d}_depth.jpg", d_vis_big)

        self.saved_count += 1

    # [NEW] 绘制并保存轨迹俯视图图表
    def save_trajectory_plot(self, frame_idx, hist_pts, fut_pts, state_vector):
        if self.saved_traj_count >= DEBUG_MAX_IMAGES: return
        
        plt.figure(figsize=(6, 8))
        
        # 1. 历史点 (蓝色)
        plt.plot(hist_pts[:, 0], hist_pts[:, 2], marker='.', color='royalblue', linewidth=2, markersize=8, label='History')
        
        # 2. 未来点 (橙红色)
        plt.plot(fut_pts[:, 0], fut_pts[:, 2], marker='^', color='orangered', linewidth=2, markersize=6, label='Future')
        
        # 3. 当前点 (黑色星号)
        curr_x, curr_z = hist_pts[-1, 0], hist_pts[-1, 2]
        plt.plot(curr_x, curr_z, marker='*', color='black', markersize=12, label='Current', zorder=4)
        
        # 4. 当前速度箭头 (绿色)
        vel_x, vel_y, vel_z = state_vector[0:3]
        speed = np.hypot(vel_x, vel_z)
        if speed > 0.1:
            plt.arrow(curr_x, curr_z, vel_x * 0.5, vel_z * 0.5, color='lime', width=0.02, head_width=0.08, zorder=5)

        # 5. 画出辅助视场角(FOV)线, 大约80度
        z_max = max(np.max(hist_pts[:, 2]), np.max(fut_pts[:, 2])) + 1.0
        tan_half_fov = math.tan(math.radians(40))
        plt.plot([0, z_max * tan_half_fov], [0, z_max], color='lightgray', linestyle='--')
        plt.plot([0, -z_max * tan_half_fov], [0, z_max], color='lightgray', linestyle='--')

        # 图表设置
        plt.xlabel("Lateral X (m)")
        plt.ylabel("Forward Z (m)")
        plt.title("Trajectory Reconstruction (Camera Frame)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axis('equal') # 保证X轴和Z轴的物理比例1:1
        
        # 动态确保对称与足够的显示范围
        max_x = max(abs(np.min(hist_pts[:, 0])), abs(np.max(fut_pts[:, 0])), 1.5)
        plt.xlim(-max_x-0.2, max_x+0.2)
        plt.ylim(min(np.min(hist_pts[:, 2])-0.5, -0.5), z_max)
        
        # 6. 左上角数据文本面板
        textstr = '\n'.join((
            f"Frame: {frame_idx:06d}",
            f"Vel_X: {vel_x:.3f} m/s",
            f"Vel_Z: {vel_z:.3f} m/s",
            f"Speed: {speed:.3f} m/s",
            f"Target Depth: {curr_z:.3f} m"
        ))
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8, edgecolor='lightgray')
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{frame_idx:06d}_traj.jpg", dpi=100)
        plt.close()
        
        self.saved_traj_count += 1
    
def determine_intention(traj_cam):
    if len(traj_cam) < 3: return 0
    v_in = traj_cam[1] - traj_cam[0]  
    v_out = traj_cam[2] - traj_cam[1] 
    
    speed_in = np.linalg.norm(v_in) / HISTORY_DT  
    speed_out = np.linalg.norm(v_out) / FUTURE_DT
    
    cross_prod = v_in[2] * v_out[0] - v_in[0] * v_out[2] 
    
    dot_prod = np.dot(v_in[[0,2]], v_out[[0,2]])
    norm_prod = np.linalg.norm(v_in[[0,2]]) * np.linalg.norm(v_out[[0,2]])
    angle = 0.0
    if norm_prod > 1e-4: angle = np.arccos(np.clip(dot_prod/norm_prod, -1.0, 1.0))
    
    if angle > math.radians(15):
        if cross_prod > 0: return 2  
        else: return 1               
        
    return 0

def process_bag_offline(bag_file_path, output_dir_path):
    print("="*60)
    print(f"Processing: {bag_file_path}")
    if not os.path.exists(bag_file_path): return
    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(f"{output_dir_path}/depth_tensor", exist_ok=True)
    os.makedirs(f"{output_dir_path}/state_vector", exist_ok=True)

    bridge = CvBridge()
    processor = DataProcessor()
    vis = Visualizer(output_dir_path) 
    yolo = YOLO('yolov8n-pose.pt')
    bag = rosbag.Bag(bag_file_path)

    processed_timeline =[]
    last_drone_pose = None; last_depth_msg = None; last_human_gt_pose = None 
    frame_idx = 0
    stats = {"total": 0, "valid": 0, "rejected_geometry": 0}
    topics_list =[RGB_TOPIC, DEPTH_TOPIC, DRONE_TOPIC, HUMAN_TOPIC]
    
    for topic, msg, t in bag.read_messages(topics=topics_list):
        if topic == DRONE_TOPIC:
            last_drone_pose = pose_msg_to_matrix(msg)
        elif topic == HUMAN_TOPIC:
            last_human_gt_pose = pose_msg_to_matrix(msg)
            if last_human_gt_pose is not None:
                p_w = last_human_gt_pose[0:3, 3]
                processor.vel_estimator.update(msg.header.stamp.to_sec(), p_w)
        elif topic == DEPTH_TOPIC:
            last_depth_msg = msg
        elif topic == RGB_TOPIC:
            stats["total"] += 1
            curr_time = msg.header.stamp.to_sec()
            
            if last_drone_pose is None or last_depth_msg is None or last_human_gt_pose is None: continue 
            if abs(curr_time - last_depth_msg.header.stamp.to_sec()) > 0.05: continue
            
            try:
                cv_rgb = bridge.imgmsg_to_cv2(msg, "bgr8")
                cv_depth = bridge.imgmsg_to_cv2(last_depth_msg, "passthrough")
                
                results = yolo(cv_rgb, verbose=False, classes=[0])
                if len(results) == 0 or len(results[0].boxes) == 0: continue
                
                r = results[0]
                box = r.boxes.xywh.cpu().numpy()[0] 
                kpts = r.keypoints.data.cpu().numpy()[0]
                
                is_valid_geometry = False
                proj_gt = None; hip_center = None
                human_pos_world = last_human_gt_pose[0:3, 3].copy()
                human_pos_world[2] -= VICON_Z_OFFSET 
                proj_gt = project_world_to_pixel(human_pos_world, last_drone_pose)
                kp_l_hip = kpts[11]; kp_r_hip = kpts[12]
                if kp_l_hip[2] > 0.3 and kp_r_hip[2] > 0.3: hip_center = (kp_l_hip[:2] + kp_r_hip[:2]) / 2.0
                else: hip_center = box[:2] 

                if proj_gt is not None:
                    dist = np.linalg.norm(proj_gt - hip_center)
                    cx, cy, w, h = box
                    in_bbox = (cx - w/2) <= proj_gt[0] <= (cx + w/2) and (cy - h/2) <= proj_gt[1] <= (cy + h/2)
                    if dist < VALIDATION_PIXEL_THRESH or in_bbox: is_valid_geometry = True
                    else: stats["rejected_geometry"] += 1
                
                centroid_cam, heading_vec_cam, sideways_flag = processor.get_centroid_and_heading(kpts, cv_depth)
                bbox_d = processor.aligner.get_aligned_depth_roi(cv_depth, box)
                depth_feat = None; roi_meta = None
                if centroid_cam is not None:
                    depth_feat, roi_meta = processor.process_depth_roi(cv_depth, bbox_d, centroid_cam[2])

                if not is_valid_geometry or centroid_cam is None or depth_feat is None: continue

                T_opt_to_world = np.dot(last_drone_pose, T_opt_to_body_mat)
                T_world_to_opt_curr = np.linalg.inv(T_opt_to_world)
                vel_cam = processor.vel_estimator.get_velocity_in_current_camera_frame(curr_time, T_world_to_opt_curr)

                if heading_vec_cam is not None:
                    vel_planar = np.array([vel_cam[0], 0.0, vel_cam[2]])
                    speed_planar = np.linalg.norm(vel_planar)
                    heading_planar = np.array([heading_vec_cam[0], 0.0, heading_vec_cam[2]])
                    heading_norm = np.linalg.norm(heading_planar)
                    
                    if speed_planar > 0.15 and heading_norm > 1e-3:
                        v_dir = vel_planar / speed_planar
                        h_dir = heading_planar / heading_norm
                        dot_prod = np.dot(h_dir, v_dir)
                        limit_cos = math.cos(math.radians(100))
                        if dot_prod < limit_cos:
                            heading_vec_cam = -heading_vec_cam

                shoulder_vis_data =[] 
                hip_vis_data = None 

                if roi_meta is not None:
                    for s_idx, label in[(5, "L"), (6, "R")]: 
                        if kpts[s_idx][2] > 0.5:
                            u_rgb, v_rgb = kpts[s_idx][:2]
                            u_d, v_d, d_val = processor.aligner.get_aligned_uv(u_rgb, v_rgb, cv_depth)
                            if u_d is not None:
                                t_x = (u_d - roi_meta['x1']) * roi_meta['scale_x']
                                t_y = (v_d - roi_meta['y1']) * roi_meta['scale_y']
                                shoulder_vis_data.append((t_x, t_y, d_val, label))
                    
                    if kpts[11][2] > 0.5 and kpts[12][2] > 0.5:
                        u_hip_rgb = (kpts[11][0] + kpts[12][0]) / 2.0
                        v_hip_rgb = (kpts[11][1] + kpts[12][1]) / 2.0
                        u_d, v_d, d_val = processor.aligner.get_aligned_uv(u_hip_rgb, v_hip_rgb, cv_depth)
                        if u_d is not None:
                            t_x = (u_d - roi_meta['x1']) * roi_meta['scale_x']
                            t_y = (v_d - roi_meta['y1']) * roi_meta['scale_y']
                            hip_vis_data = (t_x, t_y, d_val)

                if frame_idx % DEBUG_SAVE_STRIDE == 0:
                    vis.draw_and_save(
                        frame_idx, cv_rgb, box, kpts, proj_gt, hip_center, 
                        depth_feat, shoulder_vis_data, hip_vis_data,      
                        heading_vec=heading_vec_cam, sideways_flag=sideways_flag
                    )

                yaw_sin, yaw_cos = 0.0, 1.0
                if heading_vec_cam is not None:
                    yaw_sin = heading_vec_cam[0]; yaw_cos = heading_vec_cam[2]
                
                explicit_state = np.array([
                    vel_cam[0], vel_cam[1], vel_cam[2],
                    yaw_sin, yaw_cos, centroid_cam[2]
                ], dtype=np.float32)
                
                data_entry = {
                    'time': curr_time, 'file_idx': frame_idx,
                    'drone_pose': last_drone_pose, 'human_pose': last_human_gt_pose,  
                    'centroid_cam': centroid_cam, 'depth_feat': depth_feat,
                    'explicit_state': explicit_state, 'bbox': box
                }
                processed_timeline.append(data_entry)
                np.save(f"{output_dir_path}/depth_tensor/{frame_idx:06d}.npy", depth_feat)
                np.save(f"{output_dir_path}/state_vector/{frame_idx:06d}.npy", explicit_state)
                
                stats["valid"] += 1; frame_idx += 1
                print(f"Proc: {frame_idx} | Rej: {stats['rejected_geometry']} | Total: {stats['total']}", end='\r')
            except Exception as e: pass

    bag.close()
    print(f"\nProcessing Done. Total Valid: {len(processed_timeline)}")
    
    dataset_index =[]
    start_idx = HISTORY_STEPS * HISTORY_STRIDE
    end_idx = len(processed_timeline) - (FUTURE_STEPS * FUTURE_STRIDE)
    
    for i in range(start_idx, end_idx, DATA_SAMPLE_STRIDE):
        curr = processed_timeline[i]
        
        T_opt_to_world_curr = np.dot(curr['drone_pose'], T_opt_to_body_mat)
        T_world_to_opt_curr = np.linalg.inv(T_opt_to_world_curr)

        p_w_curr = curr['human_pose'][0:3, 3].copy()
        p_w_curr[2] -= VICON_Z_OFFSET
        p_opt_curr_center = transform_point(p_w_curr, T_world_to_opt_curr)

        points_seq = [p_opt_curr_center] 
        valid_hist = True
        
        for h in range(1, HISTORY_STEPS + 1):
            h_idx = i - h * HISTORY_STRIDE
            hist_item = processed_timeline[h_idx]
            
            if abs(curr['time'] - hist_item['time'] - h * HISTORY_DT) > 0.05: 
                valid_hist = False; break
            
            p_w_hist = hist_item['human_pose'][0:3, 3].copy()
            p_w_hist[2] -= VICON_Z_OFFSET
            p_opt_hist = transform_point(p_w_hist, T_world_to_opt_curr)
            points_seq.append(p_opt_hist)
            
        if not valid_hist: continue

        points_seq.reverse() 
        points_arr = np.array(points_seq)
        traj_deltas = np.diff(points_arr, axis=0)
        input_traj_deltas = traj_deltas.tolist()

        label_deltas =[] 
        valid_fut = True
        p_opt_prev = p_opt_curr_center
        p_intent_prev = points_seq[-2] 
        p_intent_curr = p_opt_curr_center
        p_intent_next = None

        # [NEW] 用于保存可视化未来点绝对坐标的集合，起点是当前位置
        fut_abs = [p_opt_curr_center]

        for f in range(1, FUTURE_STEPS + 1):
            f_idx = i + f * FUTURE_STRIDE
            fut_item = processed_timeline[f_idx]
            if abs(fut_item['time'] - curr['time'] - f * FUTURE_DT) > 0.05: valid_fut = False; break
            p_w_fut = fut_item['human_pose'][0:3, 3].copy()
            p_w_fut[2] -= VICON_Z_OFFSET
            p_opt_curr = transform_point(p_w_fut, T_world_to_opt_curr)
            
            fut_abs.append(p_opt_curr) # [NEW] 将未来点的绝对相机坐标加入用于可视化的集合

            delta = p_opt_curr - p_opt_prev
            label_deltas.append(delta.tolist())
            if f == 1: p_intent_next = p_opt_curr
            p_opt_prev = p_opt_curr 
            
        if not valid_fut: continue

        # [NEW] 在数据有效的情况下，根据条件触发轨迹画图
        if curr['file_idx'] % DEBUG_SAVE_STRIDE == 0:
            vis.save_trajectory_plot(curr['file_idx'], points_arr, np.array(fut_abs), curr['explicit_state'])
        
        intention_traj = np.array([p_intent_prev, p_intent_curr, p_intent_next])
        intention = determine_intention(intention_traj)

        dataset_index.append({
            "seq_id": f"{curr['file_idx']:06d}",
            "history_traj": input_traj_deltas,  
            "labels_delta": label_deltas,
            "intention_label": int(intention),
            "bbox": curr['bbox'].tolist()
        })
        
    with open(f"{output_dir_path}/labels.json", "w") as f:
        json.dump(dataset_index, f)
    print(f"Generated {len(dataset_index)} samples.")

if __name__ == "__main__":
    for bag_p, out_p in BAG_TASKS:
        process_bag_offline(bag_p, out_p)