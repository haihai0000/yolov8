# import json
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import torch
# import matplotlib.pyplot as plt
# import math
# import os
# def load_standard_action(json_path):
#     """
#     从 JSON 文件加载标准动作数据
#     :param json_path: JSON 文件路径
#     :return: 返回标准动作的数据
#     """
#     with open(json_path, 'r', encoding='utf-8') as f:
#         action = json.load(f)
    
#     # 提取标准动作的关键点，排除头部的关键点（鼻子，眼睛，耳朵）
#     body_keypoints_indices = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
#                               "left_wrist", "right_wrist", "left_hip", "right_hip", 
#                               "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
#     standard_keypoints = {}
    
#     # 遍历 "shapes" 获取所需的关键点
#     for shape in action['shapes']:
#         label = shape['label']
#         if label in body_keypoints_indices:
#             # 存储关键点的坐标（只考虑有效点，即 points 中的值）
#             standard_keypoints[label] = shape['points'][0]  # 只取第一个点

#     return standard_keypoints
# def rotate_pose(keypoints, angle):
#     """
#     旋转姿态，使其与标准方向对齐
#     :param keypoints: 姿态关键点 (每个关键点为 [x, y])
#     :param angle: 旋转角度 (度)
#     :return: 旋转后的关键点
#     """
#     # 将角度转为弧度
#     angle_rad = np.radians(angle)
    
#     # 计算旋转矩阵
#     rotation_matrix = np.array([
#         [np.cos(angle_rad), -np.sin(angle_rad)],
#         [np.sin(angle_rad), np.cos(angle_rad)]
#     ])
    
#     # 旋转所有关键点
#     rotated_keypoints = []
#     for kp in keypoints:
#         rotated_kp = np.dot(rotation_matrix, np.array(kp))
#         rotated_keypoints.append(rotated_kp)
    
#     return rotated_keypoints

# def normalize_pose(keypoints, ref_distance):
#     """
#     归一化姿态，调整尺度
#     :param keypoints: 姿态关键点 (每个关键点为 [x, y])
#     :param ref_distance: 参考尺度（如肩膀到膝盖的距离）
#     :return: 归一化后的关键点
#     """
#     # 计算肩膀到膝盖的距离
#     left_shoulder = np.array(keypoints[5])
#     left_knee = np.array(keypoints[11])
#     distance = np.linalg.norm(left_shoulder - left_knee)
    
#     # 计算缩放因子
#     scale_factor = ref_distance / distance
    
#     # 归一化关键点
#     normalized_keypoints = []
#     for kp in keypoints:
#         normalized_kp = np.array(kp) * scale_factor
#         normalized_keypoints.append(normalized_kp)
    
#     return normalized_keypoints

# def calculate_similarity(pose1, pose2, ref_distance=1.0):
#     """
#     计算两个姿态之间的相似度（包括旋转对齐与归一化）
#     :param pose1: 第一个姿态（关键点坐标字典）
#     :param pose2: 第二个姿态（关键点坐标字典）
#     :param ref_distance: 参考尺度（用于归一化）
#     :return: 返回姿态之间的相似度（欧几里得距离）
#     """


#     # 计算欧式距离的方法
#     '''
#     # Ensure tensors are moved to CPU before converting to numpy arrays
#     shoulder_left = np.array(pose1["left_shoulder"].cpu())  # Move to CPU
#     shoulder_right = np.array(pose1["right_shoulder"].cpu())  # Move to CPU
    
#     # 计算肩膀线的角度
#     shoulder_line = shoulder_right - shoulder_left
#     angle = np.degrees(np.arctan2(shoulder_line[1], shoulder_line[0]))  # 弧度转角度
    
#     # Handle individual keypoints to ensure tensors are moved to CPU
#     rotated_pose1 = rotate_pose([tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor for tensor in pose1.values()], -angle)
#     rotated_pose2 = rotate_pose([tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor for tensor in pose2.values()], -angle)
    
#     # 归一化姿态，使其尺寸相同
#     normalized_pose1 = normalize_pose(rotated_pose1, ref_distance)
#     normalized_pose2 = normalize_pose(rotated_pose2, ref_distance)
#     # 计算归一化后的欧几里得距离
#     distance = 0
#     for key in range(len(normalized_pose1)):
#         x1, y1 = normalized_pose1[key]
#         x2, y2 = normalized_pose2[key]
#         distance += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
#     '''

#     # 计算角度的方法
#     body_skeleton = [
#         (0, 6), (1, 7), (0, 1), (0, 2), (1, 3),  # Shoulder, Elbow, Wrist
#         (2, 4), (3, 5), (6, 8), (7, 9), (6, 7), (9, 11), (8, 10)        # Body connections from shoulders to hips and legs
#     ]
#     body_keypoints_indices = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
#                             "left_wrist", "right_wrist", "left_hip", "right_hip", 
#                             "left_knee", "right_knee", "left_ankle", "right_ankle"]
#     #filtered_keypoints = {key: keypoints[i+5] for i, key in enumerate(standard_keypoints.keys())}
#     # 提取关键点坐标
#     keypoints1 = [pose1[key] for key in body_keypoints_indices]
#     keypoints2 = [pose2[key] for key in body_keypoints_indices]

#     # 计算对应边的向量
#     def calculate_vectors(keypoints):
#         vectors = []
#         for (i, j) in body_skeleton:
#             # 计算关键点i和j之间的向量
#             x1, y1 = keypoints[i]
#             x2, y2 = keypoints[j]
#             vector = [x2-x1, y2-y1]
#             vectors.append(vector)
#         return vectors
#     def calculate_feature(vector):
#         feature = []
#         # 计算每对向量的点积除以它们的模的乘积
#         norms = np.linalg.norm(vector, axis=1) 
#         for i in range(12):
#             for j in range(i, 12):  # 从 i 开始到 12，以避免重复计算
#         # 计算第 i 和第 j 个向量的点积
#                 dot_product = np.dot(vector[i], vector[j])
        
#         # 计算它们模的乘积
#                 norm_product = norms[i] * norms[j]+0.2
        
#         # 计算点积除以模的乘积，并将结果存入 result 数组
#                 feature.append(dot_product / norm_product)
#         return feature
#     pose_vector1 = [[t1.cpu().tolist(), t2.cpu().tolist()] for t1, t2 in calculate_vectors(keypoints1)]
#     pose_vector2 = calculate_vectors(keypoints2)
#     pose_feature1 = calculate_feature(pose_vector1)
#     pose_feature2 = calculate_feature(pose_vector2)
#     result = np.dot(pose_feature1, pose_feature2)
#     if result is None or np.isnan(result):
#         import pdb
#         pdb.set_trace()
#         print("ours", pose_feature1)
#         print("standard", pose_feature2)
#     return result


# … 省略 load_standard_action、calculate_similarity 等函数 …

# def pose_detection_and_save_json(
#     video_path,
#     output_json='results.json',
#     det_conf=0.9,
#     keypoint_conf_threshold=0.5
# ):
#     # 初始化 YOLO-pose
#     model = YOLO('yolov8x-pose-p6.pt')

#     cap = cv2.VideoCapture(video_path)
#     fps    = cap.get(cv2.CAP_PROP_FPS)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     body_kp_names = [
#         "left_shoulder","right_shoulder","left_elbow","right_elbow",
#         "left_wrist","right_wrist","left_hip","right_hip",
#         "left_knee","right_knee","left_ankle","right_ankle"
#     ]
    
#     frame_idx = 0
#     all_frames = []

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # 预测
#         res = model.predict(frame, imgsz=640, conf=det_conf, device=0)[0]

#         # 选最大人体
#         sel, max_area = -1, 0
#         for i, box in enumerate(res.boxes.xyxy):
#             x1, y1, x2, y2 = map(int, box[:4])
#             area = (x2 - x1) * (y2 - y1)
#             if area > max_area:
#                 max_area, sel = area, i

#         # 准备这一帧的记录
#         info = {
#             "frame_idx": frame_idx,
#             "bbox": None,
#             "keypoints": None
#         }

#         if sel >= 0:
#             # bbox + score
#             x1, y1, x2, y2 = map(int, res.boxes.xyxy[sel][:4])
#             info["bbox"] = [x1, y1, x2, y2]

#             # keypoints 只取我们关心的 12 点
#             kpts = res.keypoints.xy[sel].cpu().numpy()
#             kp_dict = {}

#             for v, name in zip(kpts[5:17], body_kp_names):
#                 if len(v) >= 3 and v[2] >= keypoint_conf_threshold:  # 确保数组至少有3个元素
#                     kp_dict[name] = [float(v[0]), float(v[1])]

#             info["keypoints"] = kp_dict

#         all_frames.append(info)
#         frame_idx += 1

#     cap.release()

#     # 最后写到 JSON
#     out = {
#         "video_path": video_path,
#         "total_frames": frame_idx,
#         "frames": all_frames
#     }
#     with open(output_json, 'w', encoding='utf-8') as f:
#         json.dump(out, f, ensure_ascii=False, indent=2)

#     print(f"Done! Saved results to {output_json}")


# if __name__ == "__main__":
#     pose_detection_and_save_json(
#         video_path=r"C:/Users/Administrator/Desktop/person2.mp4",
#         output_json=r"C:/Users/Administrator/Desktop/frames2.json",
#         det_conf=0.5,
#         keypoint_conf_threshold=0.3
#     )



####画视频
# def pose_detection(video_path, pose_folder, output_path='output.mp4', det_conf=0.9, keypoint_conf_threshold=0.5):
#     # 初始化模型
#     model = YOLO('yolov8x-pose-p6.pt')
    
#     # 视频输入输出设置
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     body_skeleton = [
#         (0, 6), (1, 7), (0, 1), (0, 2), (1, 3),  # Shoulder, Elbow, Wrist
#         (2, 4), (3, 5), (6, 8), (7, 9), (6, 7), (9, 11), (8, 10)        # Body connections from shoulders to hips and legs
#     ]
#     # 第一遍：计算所有帧的similarity
#     frames = []  # 存储所有帧
#     similarity_distribution = []  # 存储每帧的similarity
    
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success: break
        
#         frames.append(frame)  # 保存帧
#         results = model.predict(frame, imgsz=640, conf=det_conf, device='0')[0]
        
#         # 找到最近的人
#         closest_person_idx = -1
#         max_area = 0
        
#         for idx, box in enumerate(results.boxes.xyxy):
#             x1, y1, x2, y2 = map(int, box[:4])
#             area = (x2 - x1) * (y2 - y1)
#             if area > max_area:
#                 max_area = area
#                 closest_person_idx = idx
        
#         # 如果检测到人
#         if closest_person_idx != -1:
#             keypoints = results.keypoints.xy[closest_person_idx]
#             body_keypoints_indices_temp = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
#                                          "left_wrist", "right_wrist", "left_hip", "right_hip", 
#                                          "left_knee", "right_knee", "left_ankle", "right_ankle"]
#             filtered_keypoints = {key: keypoints[i+5] for i, key in enumerate(body_keypoints_indices_temp)}
            
#             # 计算与所有标准动作的相似度总和
#             frame_similarity = 0
#             for json_file in os.listdir(pose_folder):
#                 if json_file.endswith('.json'):
#                     json_path = os.path.join(pose_folder, json_file)
#                     standard_keypoints = load_standard_action(json_path)
#                     current_similarity = calculate_similarity(filtered_keypoints, standard_keypoints)
#                     frame_similarity += current_similarity
            
#             similarity_distribution.append(frame_similarity)
#         else:
#             similarity_distribution.append(0)  # 如果没检测到人，相似度记为0
    
#     # 计算滑动窗口的相似度
#     window_size = 5  # 前后5帧加上当前帧
#     windowed_similarity = []
    
#     for i in range(len(similarity_distribution)):
#         start_idx = max(0, i - window_size//2)
#         end_idx = min(len(similarity_distribution), i + window_size//2 + 1)
#         window_sum = sum(similarity_distribution[start_idx:end_idx])
#         windowed_similarity.append(window_sum)
    
#     # —— 以下为第二遍：输出视频（替换原来对应部分） ——    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开始
#     # 修改输出视频的高度，增加图表区域
#     plot_height = height // 5  # 图表高度为原视频高度的1/5
#     new_height = height + plot_height
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, new_height))

#     # 预先计算图表的基本参数
#     max_similarity = max(windowed_similarity)
#     min_similarity = min(windowed_similarity)
#     total_frames = len(windowed_similarity)

#     body_keypoints_indices = [
#         "left_shoulder","right_shoulder","left_elbow","right_elbow",
#         "left_wrist","right_wrist","left_hip","right_hip",
#         "left_knee","right_knee","left_ankle","right_ankle"
#     ]

#     for frame_idx, frame in enumerate(frames):
#         # 新画布
#         canvas = np.zeros((new_height, width, 3), dtype=np.uint8)

#         # 1）检测并画框、画骨架
#         results = model.predict(frame, imgsz=1280, conf=det_conf, device='0')[0]
#         # 找最大人
#         closest_person_idx, max_area = -1, 0
#         for idx, box in enumerate(results.boxes.xyxy):
#             x1b, y1b, x2b, y2b = map(int, box[:4])
#             area = (x2b - x1b) * (y2b - y1b)
#             if area > max_area:
#                 max_area, closest_person_idx = area, idx

#         if closest_person_idx != -1:
#             # 边框坐标
#             x1b, y1b, x2b, y2b = map(int, results.boxes.xyxy[closest_person_idx][:4])
#             cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)

#             # 重新生成 filtered_keypoints
#             kpts = results.keypoints.xy[closest_person_idx]
#             filtered_keypoints = {
#                 key: kpts[i+5] for i, key in enumerate(body_keypoints_indices)
#             }

#             # 画骨架连线（强制 int）
#             for (i, j) in body_skeleton:
#                 x1k, y1k = filtered_keypoints[body_keypoints_indices[i]]
#                 x2k, y2k = filtered_keypoints[body_keypoints_indices[j]]
#                 # 只在都有检测到时画
#                 if x1k and y1k and x2k and y2k:
#                     cv2.line(
#                         frame,
#                         (int(x1k), int(y1k)),
#                         (int(x2k), int(y2k)),
#                         (255, 0, 0), 2
#                     )

#             # 画相似度文本，org 一定是整数元组
#             sim_val = windowed_similarity[frame_idx]
#             text = f"Similarity: {sim_val:.2f}"
#             org = (x1b, y1b - 10)                 # 用边框左上角做参考
#             cv2.putText(
#                 frame, text, org,
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9, (0, 255, 255), 2, cv2.LINE_AA
#             )

#         # 把处理后的视频帧放到 canvas 上方
#         canvas[:height, :] = frame

#         # 2）绘制下方曲线图
#         plot_area = np.ones((plot_height, width, 3), dtype=np.uint8) * 255
#         # 网格
#         for i in range(4):
#             y = int(i * plot_height / 4)
#             cv2.line(plot_area, (0, y), (width, y), (200,200,200), 1)
#         for i in range(10):
#             x = int(i * width / 10)
#             cv2.line(plot_area, (x, 0), (x, plot_height), (200,200,200), 1)

#         # 曲线点列表
#         pts = []
#         for i, sim in enumerate(windowed_similarity):
#             x = int(i * width / total_frames)
#             # 防止除零、确保 int
#             y = int(plot_height - (sim - min_similarity) * plot_height / (max_similarity - min_similarity + 1e-6))
#             pts.append((x, y))
#         # 画曲线
#         for a, b in zip(pts, pts[1:]):
#             cv2.line(plot_area, a, b, (255,0,0), 2)
#         # 当前帧红点
#         cx, cy = pts[frame_idx]
#         cv2.circle(plot_area, (cx, cy), 5, (0,0,255), -1)

#         # y 轴刻度
#         cv2.putText(plot_area, str(int(max_similarity)), (5,15),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
#         cv2.putText(plot_area, str(int(min_similarity)), (5,plot_height-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

#         # 放到 canvas 下方
#         canvas[height:, :] = plot_area

#         # 3）写入输出:自动慢放
#         # if windowed_similarity[frame_idx] >= 4500:
#         #     # 暂停帧
#         #     for _ in range(15):
#         #         cv2.putText(canvas, "Moment Pause", (x1b, y1b-20),
#         #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
#         #         out.write(canvas)
#         out.write(canvas)

#         # 预览
#         cv2.imshow('Preview', canvas)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # 使用示例
# pose_folder = r"D:/users/smash"

# pose_detection(
#     video_path=r"C:/Users\Administrator/Desktop/person1.mp4",
#     pose_folder=pose_folder,
#     output_path=r"C:/Users\Administrator/Desktop/demo.mp4",
#     det_conf=0.5,  # 目标检测置信度阈值
#     keypoint_conf_threshold=0.3  # 关键点置信度阈值
# )



######输出对应的每一帧的信息-json文件
# import json
# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO

# # ---------- 标准动作加载 ----------
# def load_standard_action(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         action = json.load(f)
#     body_kp = [
#         "left_shoulder","right_shoulder","left_elbow","right_elbow",
#         "left_wrist","right_wrist","left_hip","right_hip",
#         "left_knee","right_knee","left_ankle","right_ankle"
#     ]
#     std_kpts = {}
#     for shp in action.get('shapes', []):
#         lbl = shp.get('label')
#         if lbl in body_kp:
#             std_kpts[lbl] = shp['points'][0]
#     return std_kpts

# # ---------- 相似度计算 ----------
# def calculate_similarity(pose, standard):
#     body_skel = [(0,6),(1,7),(0,1),(0,2),(1,3),(2,4),(3,5),(6,8),(7,9),(6,7),(9,11),(8,10)]
#     idx2lbl = [
#         "left_shoulder","right_shoulder","left_elbow","right_elbow",
#         "left_wrist","right_wrist","left_hip","right_hip",
#         "left_knee","right_knee","left_ankle","right_ankle"
#     ]
#     def get_vectors(pose_dict):
#         pts = [pose_dict[lbl] for lbl in idx2lbl]
#         vecs = []
#         for i, j in body_skel:
#             x1, y1 = pts[i]; x2, y2 = pts[j]
#             vecs.append([x2-x1, y2-y1])
#         return np.array(vecs)

#     v1 = get_vectors(pose)
#     v2 = get_vectors(standard)
#     norms1 = np.linalg.norm(v1, axis=1) + 1e-6
#     norms2 = np.linalg.norm(v2, axis=1) + 1e-6

#     feat1, feat2 = [], []
#     for a in range(len(v1)):
#         for b in range(a, len(v1)):
#             feat1.append(np.dot(v1[a], v1[b]) / (norms1[a] * norms1[b]))
#             feat2.append(np.dot(v2[a], v2[b]) / (norms2[a] * norms2[b]))
#     feat1 = np.array(feat1)
#     feat2 = np.array(feat2)
#     return float(np.dot(feat1, feat2))

# # ---------- 主函数：提取姿态+相似度 ----------
# def extract_with_similarity(video_path, pose_folder, output_json,
#                             det_conf=0.9, device=0):
#     # 预加载标准动作
#     std_actions = []
#     for fn in os.listdir(pose_folder):
#         if fn.lower().endswith('.json'):
#             std_actions.append(load_standard_action(os.path.join(pose_folder, fn)))

#     model = YOLO('yolov8x-pose-p6.pt')
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open {video_path}")

#     data = []
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         res = model.predict(frame, imgsz=640, conf=det_conf, device=device)[0]
#         max_area, sel = 0, -1
#         for i, box in enumerate(res.boxes.xyxy):
#             x1,y1,x2,y2 = map(int, box[:4])
#             area = (x2-x1)*(y2-y1)
#             if area > max_area:
#                 max_area, sel = area, i

#         info = {
#             "frame_idx": frame_idx,
#             "bbox": None,
#             "keypoints": None,
#             "score": None,
#             "similarity": None
#         }

#         if sel >= 0:
#             box = res.boxes.xyxy[sel]; conf = res.boxes.conf[sel].item()
#             info["bbox"] = list(map(int, box[:4]))
#             info["score"] = float(conf)

#             kpts = res.keypoints.xy[sel].cpu().numpy()
#             labels = [
#                 "left_shoulder","right_shoulder","left_elbow","right_elbow",
#                 "left_wrist","right_wrist","left_hip","right_hip",
#                 "left_knee","right_knee","left_ankle","right_ankle"
#             ]
#             kp_dict = {labels[i]: kpts[i+5].tolist() for i in range(12)}
#             info["keypoints"] = kp_dict

#             sim_sum = 0.0
#             for std in std_actions:
#                 sim_sum += calculate_similarity(kp_dict, std)
#             info["similarity"] = sim_sum

#         data.append(info)
#         frame_idx += 1

#     cap.release()
#     with open(output_json, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     print(f"Saved {len(data)} frames (with similarity) to {output_json}")

# # ---------- 硬编码参数区域 ----------
# video_path    = r"C:/Users/Administrator/Desktop/person2.mp4"
# pose_folder   = r"D:/users/smash"
# output_json   = r"C:/Users/Administrator/Desktop/frames2.json"
# det_conf      = 0.5  # 目标检测置信度阈值
# device        = 0    # 计算设备 (0=CPU/GPU)

# extract_with_similarity(
#     video_path,
#     pose_folder,
#     output_json,
#     det_conf=det_conf,
#     device=device
# )



import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- 标准动作加载 ----------
def load_standard_action(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        action = json.load(f)
    body_kp = [
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]
    std_kpts = {}
    for shp in action.get('shapes', []):
        lbl = shp.get('label')
        if lbl in body_kp:
            std_kpts[lbl] = shp['points'][0]
    return std_kpts

# ---------- 相似度计算 ----------
def calculate_similarity(pose, standard):
    body_skel = [(0,6),(1,7),(0,1),(0,2),(1,3),(2,4),(3,5),(6,8),(7,9),(6,7),(9,11),(8,10)]
    idx2lbl = [
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]
    def get_vectors(pose_dict):
        pts = [pose_dict[lbl] for lbl in idx2lbl]
        vecs = []
        for i, j in body_skel:
            x1, y1 = pts[i]; x2, y2 = pts[j]
            vecs.append([x2-x1, y2-y1])
        return np.array(vecs)

    v1 = get_vectors(pose)
    v2 = get_vectors(standard)
    norms1 = np.linalg.norm(v1, axis=1) + 1e-6
    norms2 = np.linalg.norm(v2, axis=1) + 1e-6

    feat1, feat2 = [], []
    for a in range(len(v1)):
        for b in range(a, len(v1)):
            feat1.append(np.dot(v1[a], v1[b]) / (norms1[a] * norms1[b]))
            feat2.append(np.dot(v2[a], v2[b]) / (norms2[a] * norms2[b]))
    feat1 = np.array(feat1)
    feat2 = np.array(feat2)
    return float(np.dot(feat1, feat2))

# ---------- 主函数：提取姿态+相似度 ----------
def extract_with_similarity(video_path, pose_folder, output_json,
                            det_conf=0.9, device=0):
    # 预加载标准动作
    std_actions = []
    for fn in os.listdir(pose_folder):
        if fn.lower().endswith('.json'):
            std_actions.append(load_standard_action(os.path.join(pose_folder, fn)))

    # 打开视频，取原始宽高
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO('yolo11x-pose.pt')

    data_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(frame, imgsz=640, conf=det_conf, device=device)[0]
        # 只选最大的人体框
        max_area, sel = 0, -1
        for i, box in enumerate(res.boxes.xyxy):
            x1,y1,x2,y2 = map(int, box[:4])
            area = (x2-x1)*(y2-y1)
            if area > max_area:
                max_area, sel = area, i

        info = {
            "frame_idx": frame_idx,
            "bbox": None,
            "keypoints": None,
            "score": None,
            "similarity": None
        }

        if sel >= 0:
            box = res.boxes.xyxy[sel]; conf = res.boxes.conf[sel].item()
            info["bbox"]  = list(map(int, box[:4]))
            info["score"] = float(conf)

            kpts = res.keypoints.xy[sel].cpu().numpy()
            labels = [
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ]
            kp_dict = {labels[i]: kpts[i+5].tolist() for i in range(12)}
            info["keypoints"] = kp_dict

            # 累加所有标准动作的相似度
            sim_sum = 0.0
            for std in std_actions:
                sim_sum += calculate_similarity(kp_dict, std)
            info["similarity"] = sim_sum

        data_frames.append(info)
        frame_idx += 1

    cap.release()

    # 最终输出：先把宽高写进去，再写 frames 列表
    out = {
        "source_size": {"width": orig_w, "height": orig_h},
        "frames": data_frames
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data_frames)} frames (with similarity) to {output_json}")

# ---------- 硬编码参数区域 ----------
if __name__ == "__main__":
    video_path    = r"C:/Users/Administrator/Desktop/person2.mp4"
    pose_folder   = r"D:/users/smash"
    output_json   = r"C:/Users/Administrator/Desktop/frames2.json"
    det_conf      = 0.5
    device        = 0
    extract_with_similarity(
        video_path,
        pose_folder,
        output_json,
        det_conf=det_conf,
        device=device
    )
