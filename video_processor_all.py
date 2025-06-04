import os
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import shutil

def process_video(region_points, video_path, model_path='/ssd1/zq/ultralytics-8.3.105/runs/pose/train11/weights/best.pt', output_dir='output', conf_threshold=0.5, device='cuda:0'):
    """
    处理视频，检测区域内的人体姿态，并保存结果
    
    参数:
    region_points: 区域多边形的点坐标列表，如 [(x1,y1), (x2,y2), ...]
    video_path: 输入视频的路径
    model_path: YOLOv8姿态估计模型的路径
    output_dir: 输出目录
    conf_threshold: 置信度阈值
    device: 使用的设备 ('cuda:0', 'cpu' 等)
    
    返回:
    output_video_path: 输出视频的路径
    output_json_path: 输出JSON文件的路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    model.to(device)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
    
    # 准备输出视频
    output_video_path = os.path.join(output_dir, 'output_video.mp4')
    output_json_path = os.path.join(output_dir, 'detections.json')
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 准备JSON数据
    json_data = {
        'video_info': {
            'path': video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames
        },
        'region': region_points,
        'frames': []
    }
    
    # 关键点名称 (COCO格式)
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # 判断点是否在区域内的函数
    def is_point_in_region(point, region):
        """判断点是否在多边形区域内"""
        x, y = point
        n = len(region)
        inside = False
        p1x, p1y = region[0]
        for i in range(1, n + 1):
            p2x, p2y = region[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    # 判断关键点是否在区域内的函数
    def is_keypoint_in_region(keypoint, region):
        """判断关键点是否在区域内"""
        x, y, conf = keypoint[0], keypoint[1], keypoint[2]
        # 如果关键点置信度太低，则认为不在区域内
        if conf < conf_threshold:
            return False
        return is_point_in_region((x, y), region)
    
    # 判断目标是否在区域内的函数
    def is_box_in_region(bbox, keypoints, region):
        """判断目标是否在区域内，使用右脚踝关键点"""
        # 如果有关键点信息，使用右脚踝关键点
        if keypoints is not None and len(keypoints) >= 17:  # 假设有17个关键点
            right_ankle = keypoints[16]  # 右脚踝是第17个关键点(索引16)
            return is_keypoint_in_region(right_ankle, region)
        
        # 如果没有关键点信息或右脚踝置信度太低，回退到使用边界框底部中点
        x1, y1, x2, y2 = bbox
        bottom_center = ((x1 + x2) / 2, y2)
        return is_point_in_region(bottom_center, region)
    
    # 处理每一帧
    frame_idx = 0
    region_np = np.array(region_points, dtype=np.int32)
    
    try:
        with tqdm(total=total_frames, desc="处理视频") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存原始帧
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # 绘制区域
                cv2.polylines(frame, [region_np], True, (0, 255, 0), 2)
                
                # 准备当前帧的JSON数据
                frame_data = {
                    'frame_id': frame_idx,
                    'frame_path': frame_path,
                    'detections': []
                }
                
                # 使用YOLOv8进行检测
                results = model.predict(frame, conf=conf_threshold, device=device)
                
                # 处理检测结果
                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    keypoints = result.keypoints if hasattr(result, 'keypoints') else None
                    
                    # 处理每个检测结果
                    for i, box in enumerate(boxes):
                        # 获取边界框信息
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        conf = box.conf.cpu().numpy()[0]
                        cls_id = int(box.cls.cpu().numpy()[0])
                        
                        # 获取关键点
                        kpts = None
                        if keypoints is not None:
                            try:
                                kpts = keypoints[i].data.cpu().numpy()
                                
                                # 确保关键点格式正确
                                if kpts.ndim == 3:  # 如果是三维数组 [1, N, 3]
                                    kpts = kpts[0]  # 取第一个维度
                                elif kpts.ndim != 2 or kpts.shape[1] != 3:  # 如果不是 [N, 3] 格式
                                    print(f"警告: 关键点格式不正确: {kpts.shape}")
                                    if kpts.ndim == 1 and len(kpts) % 3 == 0:
                                        kpts = kpts.reshape(-1, 3)
                                    else:
                                        kpts = None
                            except Exception as e:
                                print(f"处理关键点时出错: {e}")
                                print(f"关键点数据: {keypoints[i].data if keypoints is not None else 'None'}")
                                kpts = None
                        
                        # 检查是否在区域内
                        if is_box_in_region([x1, y1, x2, y2], kpts, region_points):
                            # 添加到JSON数据
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_id': int(cls_id),
                                'keypoints': []
                            }
                            
                            # 添加关键点
                            if kpts is not None:
                                for kid in range(len(kpts)):
                                    try:
                                        kpt = kpts[kid]
                                        x, y, kpt_conf = kpt[0], kpt[1], kpt[2]
                                        detection['keypoints'].append({
                                            'name': keypoint_names[kid] if kid < len(keypoint_names) else f"kpt_{kid}",
                                            'x': float(x),
                                            'y': float(y),
                                            'confidence': float(kpt_conf)
                                        })
                                    except Exception as e:
                                        print(f"处理关键点 {kid} 时出错: {e}")
                                        print(f"关键点数据类型: {type(kpts)}, 形状: {kpts.shape if hasattr(kpts, 'shape') else 'no shape'}")
                                        if kid < len(kpts):
                                            print(f"问题关键点: {kpts[kid]}")
                            
                            frame_data['detections'].append(detection)
                            
                            # 绘制边界框 - 蓝色
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # 绘制类别和置信度
                            cv2.putText(frame, f"{cls_id} {conf:.2f}", (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                            # 绘制关键点
                            if kpts is not None:
                                # 关键点连接关系 (COCO格式)
                                skeleton = [
                                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                                    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                                    [2, 4], [3, 5], [4, 6], [5, 7]
                                ]
                                
                                # 绘制关键点
                                for kid in range(len(kpts)):
                                    try:
                                        kpt = kpts[kid]
                                        x, y, kpt_conf = kpt[0], kpt[1], kpt[2]
                                        if kpt_conf > conf_threshold:  # 只绘制置信度高的关键点
                                            x, y = int(x), int(y)
                                            # 绘制关键点
                                            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                                            
                                            # 绘制关键点置信度
                                            kpt_name = keypoint_names[kid] if kid < len(keypoint_names) else f"kpt_{kid}"
                                            cv2.putText(frame, f"{kpt_name}: {kpt_conf:.2f}", 
                                                       (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                                       0.4, (0, 255, 255), 1)
                                    except Exception as e:
                                        print(f"绘制关键点 {kid} 时出错: {e}")
                                        print(f"关键点数据: {kpts[kid] if kid < len(kpts) else 'index out of range'}")
                                
                                # 绘制骨架
                                for sk_id, sk in enumerate(skeleton):
                                    try:
                                        if sk[0]-1 < len(kpts) and sk[1]-1 < len(kpts):
                                            pos1 = (int(kpts[sk[0]-1][0]), int(kpts[sk[0]-1][1]))
                                            pos2 = (int(kpts[sk[1]-1][0]), int(kpts[sk[1]-1][1]))
                                            if kpts[sk[0]-1][2] > conf_threshold and kpts[sk[1]-1][2] > conf_threshold:
                                                cv2.line(frame, pos1, pos2, (255, 0, 0), 2)
                                    except Exception as e:
                                        print(f"绘制骨架 {sk} 时出错: {e}")
                                        print(f"关键点索引: {sk[0]-1}, {sk[1]-1}, 关键点总数: {len(kpts)}")
                
                # 保存处理后的帧
                result_path = os.path.join(results_dir, f"result_{frame_idx:06d}.jpg")
                cv2.imwrite(result_path, frame)
                
                # 写入视频
                out.write(frame)
                
                # 添加帧数据到JSON
                json_data['frames'].append(frame_data)
                
                # 直接输出当前帧的JSON数据
                print(f"\n--- 帧 {frame_idx} JSON数据 ---")
                print(json.dumps(frame_data, indent=2))
                
                # 每5帧保存一次当前的JSON数据，防止程序中断导致数据丢失
                if frame_idx % 5 == 0:
                    temp_json_path = os.path.join(output_dir, 'detections_temp.json')
                    with open(temp_json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                
                frame_idx += 1
                pbar.update(1)
    
    finally:
        # 释放资源
        cap.release()
        out.release()
        
        # 保存JSON数据
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n处理完成! 共处理 {frame_idx} 帧")
        print(f"输出视频保存至: {output_video_path}")
        print(f"检测结果保存至: {output_json_path}")
    
    return output_video_path, output_json_path

def main():
    # 示例用法
    # 定义区域 - 这里使用示例区域，请根据实际情况修改
    region = [(103, 754), (996, 619), (1771, 727), (1709, 1078), (62, 1078), (102, 753)]
    
    # 视频路径
    video_path = "/ssd1/zq/ultralytics-8.3.105/video-test/1.MP4"  # 请替换为实际视频路径
    
    # 模型路径
    model_path = "/ssd1/zq/ultralytics-8.3.105/runs/pose/train11/weights/best.pt"  # 请替换为实际模型路径
    
    # 处理视频
    output_video, output_json = process_video(
        region_points=region,
        video_path=video_path,
        model_path=model_path,
        output_dir="video_output",
        conf_threshold=0.5,
        device="cuda:0"
    )
    
    print(f"视频处理完成!")
    print(f"输出视频: {output_video}")
    print(f"输出JSON: {output_json}")

if __name__ == "__main__":
    main()
