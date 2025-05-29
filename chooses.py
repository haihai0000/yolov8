
# import os
# import cv2
# import numpy as np
# from shapely.geometry import Polygon
# from ultralytics import YOLO


# VIDEO_PATH = r"C:/Users/Administrator/Desktop/演示0.mp4"
# OUTPUT_PATH = r"C:/Users/Administrator/Desktop/pose1.mp4"


# # Define body skeleton connections and keypoint labels
# BODY_SKELETON = [
#      (5, 7), (7, 9),     # 左臂
#     (6, 8), (8, 10),    # 右臂
#     (5, 6),             # 双肩
#     (5, 11), (6, 12),   # 躯干
#     (11, 12),           # 臀部
#     (11, 13), (13, 15), # 左腿
#     (12, 14), (14, 16)  # 右腿
# ]
# KEYPOINT_LABELS = [
#     "nose", "left_eye", "right_eye", "left_ear", "right_ear",
#     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
#     "left_wrist", "right_wrist", "left_hip", "right_hip",
#     "left_knee", "right_knee", "left_ankle", "right_ankle"
# ]


# def detect_two_largest_persons(video_path, output_path='output.mp4', det_conf=0.5, imgsz=640, device='0'):
#     """
#     Detect and draw bounding boxes and keypoints for the two largest persons per frame in a video.
#     :param video_path: Path to input video file.
#     :param output_path: Path to save the output video with annotations.
#     :param det_conf: Detection confidence threshold.
#     :param imgsz: Inference image size.
#     :param device: Device ID for inference (e.g., '0' for GPU).
#     """
#     # Initialize YOLO pose model
#     model = YOLO('yolov8x-pose-p6.pt')

#     # Open video capture
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video: {video_path}")

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Setup video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run pose detection
#         results = model.predict(frame, imgsz=imgsz, conf=det_conf, device=device)[0]

#         # Collect areas and indices for all detected persons
#         areas = []
#         for idx, box in enumerate(results.boxes.xyxy):
#             x1, y1, x2, y2 = map(int, box[:4])
#             area = (x2 - x1) * (y2 - y1)
#             areas.append((area, idx))

#         # Sort by area descending and pick up to two
#         top_two = sorted(areas, key=lambda x: x[0], reverse=True)[:2]
#         selected_idxs = [idx for _, idx in top_two]

#         # Draw annotations for each selected person
#         for idx in selected_idxs:
#             # Bounding box
#             x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx][:4])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Keypoints
#             kpts = results.keypoints.xy[idx]
#             kpt_confs = results.keypoints.conf[idx]

#             # Draw keypoints and skeleton
#             for kp_idx, (kp, conf) in enumerate(zip(kpts, kpt_confs)):
#                 xk, yk = int(kp[0]), int(kp[1])
#                 if conf >= 0.5:
#                     # Draw circle for keypoint
#                     cv2.circle(frame, (xk, yk), 4, (0, 0, 255), -1)
#                     # Optionally label
#                     cv2.putText(frame, KEYPOINT_LABELS[kp_idx], (xk + 2, yk - 2),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

#             # Draw skeleton connections
#             for (i, j) in BODY_SKELETON:
#                 xi, yi = kpts[i]
#                 xj, yj = kpts[j]
#                 if kpt_confs[i] >= 0.5 and kpt_confs[j] >= 0.5:
#                     cv2.line(frame,
#                              (int(xi), int(yi)),
#                              (int(xj), int(yj)),
#                              (255, 0, 0), 2)

#         # Write annotated frame
#         out.write(frame)

#         # Optional: display (comment out for headless)
#         # cv2.imshow('Pose Detection', frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()



# detect_two_largest_persons(
#     VIDEO_PATH,
#     OUTPUT_PATH,
#     det_conf=0.5,
#     imgsz=640,
#     device='0'
# )


#####为0的不连线
# import json
# import cv2

# # 连接骨架的索引对（对应 JSON 中 keypoints 的顺序）
# body_skeleton = [
#     (0, 6), (1, 7), (0, 1), (0, 2), (1, 3),
#     (2, 4), (3, 5), (6, 8), (7, 9), (6, 7),
#     (9, 11), (8, 10)
# ]
# body_keypoints_indices = [
#     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
#     "left_wrist", "right_wrist", "left_hip", "right_hip",
#     "left_knee", "right_knee", "left_ankle", "right_ankle"
# ]

# def overlay_json_on_video(json_path, input_video, output_video):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         frames_info = json.load(f)

#     cap = cv2.VideoCapture(input_video)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video {input_video}")

#     fps    = cap.get(cv2.CAP_PROP_FPS)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

#     for frame_idx in range(len(frames_info)):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         info = frames_info[frame_idx]
#         if info.get("bbox") is not None:
#             x1, y1, x2, y2 = info["bbox"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

#             # 写相似度
#             sim = info.get("similarity", 0)
#             sim_text = f"Sim: {sim:.2f}"
#             text_x = x1
#             text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
#             cv2.putText(frame, sim_text, (text_x, text_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#             kpts = info["keypoints"]
#             # 先画骨架连线，但过滤掉 (0, 0) 点
#             for idx_start, idx_end in body_skeleton:
#                 lbl1 = body_keypoints_indices[idx_start]
#                 lbl2 = body_keypoints_indices[idx_end]
#                 if lbl1 in kpts and lbl2 in kpts:
#                     x_start, y_start = kpts[lbl1]
#                     x_end,   y_end   = kpts[lbl2]
#                     # 只有当两个点都不是 (0, 0) 时才连线
#                     if (x_start, y_start) != (0, 0) and (x_end, y_end) != (0, 0):
#                         cv2.line(frame, (int(x_start), int(y_start)),
#                                         (int(x_end),   int(y_end)),
#                                         (255,0,0), 2)
#             # 再画所有非零关键点
#             for lbl, pt in kpts.items():
#                 x, y = pt
#                 if (x, y) != (0, 0):
#                     cv2.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)

#             # 可选：写置信度
#             score = info.get("score", 0)
#             cv2.putText(frame, f"Score: {score:.2f}", (x1, y2 + 25),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

#         out.write(frame)

#     cap.release()
#     out.release()
#     print(f"Overlay complete, saved to {output_video}")



# if __name__ == "__main__":
#     json_path    = r"C:/Users/Administrator/Desktop/frames2.json"
#     input_video  = r"C:/Users/Administrator/Desktop/演示0.mp4"
#     output_video = r"C:/Users/Administrator/Desktop/person1_j3.mp4"
#     overlay_json_on_video(json_path, input_video, output_video)


# import json

# def find_zero_keypoint_frames(json_path):
#     """
#     从 JSON 文件中找出含有关键点坐标为 0 的帧索引。
#     返回一个按升序排列的帧索引列表。
#     """
#     with open(json_path, 'r', encoding='utf-8') as f:
#         frames = json.load(f)

#     zero_frames = []
#     for frame in frames:
#         kpts = frame.get("keypoints")
#         # 如果没有 keypoints 跳过
#         if not kpts:
#             continue

#         # 检查是否有任意坐标为 0
#         for pt in kpts.values():
#             x, y = pt
#             if x == 0 or y == 0:
#                 zero_frames.append(frame["frame_idx"])
#                 break

#     return sorted(zero_frames)

# if __name__ == "__main__":
#     json_path = r"C:/Users/Administrator/Desktop/frames1.json"
#     zero_frame_indices = find_zero_keypoint_frames(json_path)
#     print("关键点坐标为 0 的帧索引：", zero_frame_indices)



# import json
# import cv2

# # 连接骨架的索引对（对应 keypoints 顺序）
# body_skeleton = [
#     (0, 6), (1, 7), (0, 1), (0, 2), (1, 3),
#     (2, 4), (3, 5), (6, 8), (7, 9), (6, 7),
#     (9, 11), (8, 10)
# ]
# body_keypoints_indices = [
#     "left_shoulder","right_shoulder","left_elbow","right_elbow",
#     "left_wrist","right_wrist","left_hip","right_hip",
#     "left_knee","right_knee","left_ankle","right_ankle"
# ]

# def overlay_json_on_video(json_path, input_video, output_video):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         js = json.load(f)
#     src_w = js["source_size"]["width"]
#     src_h = js["source_size"]["height"]
#     frames_info = js["frames"]

#     cap = cv2.VideoCapture(input_video)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video {input_video}")

#     # 目标视频的宽高，用来算缩放比例
#     tgt_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     tgt_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = cap.get(cv2.CAP_PROP_FPS)

#     scale_x = tgt_w / src_w
#     scale_y = tgt_h / src_h

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_video, fourcc, fps, (tgt_w, tgt_h))

#     for info in frames_info:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if info.get("bbox") is not None:
#             # 缩放 bbox
#             x1, y1, x2, y2 = info["bbox"]
#             x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
#             x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

#             # 写相似度
#             sim = info.get("similarity", 0)
#             sim_text = f"Sim: {sim:.2f}"
#             ty = y1-10 if y1>20 else y1+20
#             cv2.putText(frame, sim_text, (x1, ty),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#             # 缩放并画骨架
#             kpts = info["keypoints"]
#             # 先连线
#             for i,j in body_skeleton:
#                 lbl1 = body_keypoints_indices[i]
#                 lbl2 = body_keypoints_indices[j]
#                 if lbl1 in kpts and lbl2 in kpts:
#                     x0,y0 = kpts[lbl1]; x1_,y1_ = kpts[lbl2]
#                     if (x0,y0)!=(0,0) and (x1_,y1_)!=(0,0):
#                         x0 = int(x0 * scale_x); y0 = int(y0 * scale_y)
#                         x1_ = int(x1_ * scale_x); y1_ = int(y1_ * scale_y)
#                         cv2.line(frame, (x0,y0), (x1_,y1_), (255,0,0), 2)
#             # 再画关键点
#             for lbl, pt in kpts.items():
#                 x,y = pt
#                 if (x,y)!=(0,0):
#                     xi = int(x * scale_x); yi = int(y * scale_y)
#                     cv2.circle(frame, (xi, yi), 4, (0,0,255), -1)

#             # 写置信度
#             score = info.get("score", 0)
#             cv2.putText(frame, f"Score: {score:.2f}", (x1, y2+25),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

#         out.write(frame)

#     cap.release()
#     out.release()
#     print(f"Overlay complete, saved to {output_video}")

# if __name__ == "__main__":
#     json_path    = r"C:/Users/Administrator/Desktop/frames2.json"
#     # 如果想在同一视频上画，就直接用 person2.mp4
#     # input_video  = r"C:/Users/Administrator/Desktop/person2.mp4"
#     # 如果确实要画到另一个视频，就填演示0.mp4
#     input_video  = r"C:/Users/Administrator/Desktop/person1_j2.mp4"
#     output_video = r"C:/Users/Administrator/Desktop/person1_j3.mp4"
#     overlay_json_on_video(json_path, input_video, output_video)

################计算目录下视频总和##################
import os
import subprocess

def get_video_duration(path):
    """
    调用 ffprobe 返回视频时长（秒，float）。
    """
    # -v error：只输出错误信息  
    # -show_entries format=duration：只输出时长  
    # -of default=noprint_wrappers=1:nokey=1：简洁输出纯数字  
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return float(output.strip())

def sum_durations(root_dir, exts=(".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv")):
    total_seconds = 0.0
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                fullpath = os.path.join(dirpath, fn)
                try:
                    dur = get_video_duration(fullpath)
                    total_seconds += dur
                    print(f"{fn} —— {dur:.2f}s")
                except Exception as e:
                    print(f"无法读取 {fn}：{e}")
    return total_seconds

if __name__ == "__main__":
    root = r"Z:/BadData/Video/Amateur/double/left"
    total_s = sum_durations(root)
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    print(f"\n总时长：{total_s:.2f} 秒，约 {h} 时 {m} 分 {s:.2f} 秒")

################计算三个运动员视频总和#####################
# import os
# import subprocess

# # —— 配置区域 —— #
# # 三位运动员文件夹的路径
# PLAYERS = {
#     "A_2": r"Z:\BadData\Action\A_2",
#     "A_3": r"Z:\BadData\Action\A_3",
#     "A_4": r"Z:\BadData\Action\A_4",
# }

# # 支持的影片后缀
# VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv")
# # —— 配置结束 —— #

# def get_video_duration(path):
#     """
#     调用 ffprobe 返回视频时长（秒，float）。
#     需保证系统已安装 ffprobe 并在 PATH 中可用。
#     """
#     cmd = [
#         "ffprobe", "-v", "error",
#         "-show_entries", "format=duration",
#         "-of", "default=noprint_wrappers=1:nokey=1",
#         path
#     ]
#     out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
#     return float(out.strip())

# def sum_durations(root_dir):
#     """
#     遍历 root_dir 下所有子目录，累加所有 VIDEO_EXTS 中的文件时长（秒）。
#     """
#     total = 0.0
#     for dirpath, _, filenames in os.walk(root_dir):
#         for fn in filenames:
#             if fn.lower().endswith(VIDEO_EXTS):
#                 full = os.path.join(dirpath, fn)
#                 try:
#                     d = get_video_duration(full)
#                     total += d
#                 except Exception as e:
#                     print(f"⚠️ 读取失败：{full} —— {e}")
#     return total

# def format_hms(seconds):
#     h = int(seconds // 3600)
#     m = int((seconds % 3600) // 60)
#     s = seconds % 60
#     return f"{h}h {m}m {s:.2f}s"

# if __name__ == "__main__":
#     per_player = {}
#     grand_total = 0.0

#     for name, path in PLAYERS.items():
#         print(f"计算 {name} …")
#         t = sum_durations(path)
#         per_player[name] = t
#         grand_total += t
#         print(f"  ➤ {name} 总时长：{t:.2f}s （{format_hms(t)}）\n")

#     print("====== 全部运动员总时长 ======")
#     print(f"  ➤ 合计：{grand_total:.2f}s （{format_hms(grand_total)}）")

###################对抽帧的图片文件夹进行去除冗余帧#######################
# import os
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# def deduplicate_by_ssim(frame_dir, out_dir, threshold=0.95):
#     """
#     参数：
#       frame_dir (str)：原始帧文件夹（已按时间或文件名排序）
#       out_dir   (str)：去重后帧输出路径
#       threshold (float)：SSIM 阈值，越接近 1 去重越严格

#     输出：
#       只把“与上一保留帧”SSIM < threshold 的帧拷贝到 out_dir
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
#     filenames = sorted([f for f in os.listdir(frame_dir)
#                         if os.path.splitext(f.lower())[1] in exts])
#     prev_gray = None
#     kept = 0

#     for fname in filenames:
#         img_path = os.path.join(frame_dir, fname)
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"警告：无法读取 {img_path}，已跳过。")
#             continue

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         if prev_gray is None:
#             # 第一帧必留
#             cv2.imwrite(os.path.join(out_dir, fname), img)
#             prev_gray = gray
#             kept += 1
#             continue

#         score = ssim(prev_gray, gray)
#         if score < threshold:
#             # 差别足够大，保留
#             cv2.imwrite(os.path.join(out_dir, fname), img)
#             prev_gray = gray
#             kept += 1

#     print(f"总共保留 {kept} 帧（阈值 {threshold}）")


# if __name__ == "__main__":
#     # —— 硬编码区 —— #
#     frame_dir = r"Z:/BadData/Video/Amateur/double/left/1"
#     out_dir   = r"Z:/BadData/Video/Amateur/double/left_clean/1"
#     threshold = 0.92   # 调整去重严格度：越接近 1 去重越严格
#     # ———————— #

#     deduplicate_by_ssim(frame_dir, out_dir, threshold)

####################抽取关键帧#################
# import cv2
# import numpy as np
# import os

# def keyframes_by_hist(frame_dir, out_dir, bins=(8,8,8), diff_thresh=0.3):
#     """
#     参数：
#       frame_dir  ：原始帧目录（按文件名排序）
#       out_dir    ：关键帧输出目录
#       bins       ：HSV 直方图各通道的 bin 数
#       diff_thresh：归一化直方图差异阈值（0–1）
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     filenames = sorted(os.listdir(frame_dir))
#     prev_hist = None
#     kept = 0

#     for fname in filenames:
#         img_path = os.path.join(frame_dir, fname)
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"警告：无法读取 {img_path}，跳过。")
#             continue

#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0,1,2], None, bins,
#                             [0,180, 0,256, 0,256])
#         cv2.normalize(hist, hist)
#         hist_flat = hist.flatten()

#         if prev_hist is None:
#             # 第一帧必留
#             cv2.imwrite(os.path.join(out_dir, fname), img)
#             prev_hist = hist_flat
#             kept += 1
#             continue

#         # 用 Bhattacharyya 距离度量差异
#         diff = cv2.compareHist(prev_hist.astype('float32'),
#                                hist_flat.astype('float32'),
#                                cv2.HISTCMP_BHATTACHARYYA)
#         if diff > diff_thresh:
#             cv2.imwrite(os.path.join(out_dir, fname), img)
#             prev_hist = hist_flat
#             kept += 1

#     print(f"总共提取 {kept} 个关键帧（阈值 {diff_thresh}）")


# if __name__ == "__main__":
#     # —— 在这里硬编码你的路径和参数 —— #
#     frame_dir = r"Z:/BadData/Video/Amateur/double/left/1"
#     out_dir   = r"Z:/BadData/Video/Amateur/double/left_clean/1"
#     bins       = (8, 8, 8)   # HSV 三通道的直方图分桶数
#     diff_thresh = 0.3         # 直方图差异阈值
#     # ———————————————————————— #

#     keyframes_by_hist(frame_dir, out_dir, bins, diff_thresh)


###################pt换oonx格式
# export_to_onnx.py


# from ultralytics import YOLO

# model = YOLO("yolo11x-pose.pt")
# onnx_path = model.export(
#     format="onnx",       # 指定 ONNX 格式
#     opset=15,            # 可选：指定 opset 版本
#     simplify=True        # 可选：简化模型
# )
# print(f"导出的 ONNX 文件保存为: {onnx_path}")

