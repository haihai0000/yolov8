import os
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import shutil

def process_video(region_points, video_path, model_path, output_dir='output', conf_threshold=0.5, device='cuda:0'):
    """
    处理视频，检测区域内的人体姿态，并返回指定格式的字典
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
    
    # 准备输出字典
    data = {
        "video_dict": os.path.abspath(output_dir),
        "num_frames": total_frames,
        "motions_info": []
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
    
    # 修改后的区域判断函数
    def is_box_in_region(bbox, keypoints, region, conf_threshold):
        """判断目标是否在区域内，使用右脚踝关键点"""
        if keypoints is not None and len(keypoints) >= 17:
            try:
                right_ankle = keypoints[16]
                x, y = right_ankle[0], right_ankle[1]
                conf = right_ankle[2]  # 现在可以安全访问索引2
                if conf >= conf_threshold:
                    return is_point_in_region((x, y), region)
            except IndexError as e:
                print(f"关键点格式错误: {e}")
                pass
        
        # 回退到边界框底部中点
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
                frame_name = f"{frame_idx}.jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                
                # 准备当前帧数据
                frame_data = {
                    "frame": frame_name,
                    "detections": []
                }
                
                # 使用YOLOv8进行检测
                results = model.predict(frame, conf=conf_threshold, device=device)
                
                # 处理检测结果
                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    keypoints = result.keypoints if hasattr(result, 'keypoints') else None
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        conf = box.conf.cpu().numpy()[0]
                        cls_id = int(box.cls.cpu().numpy()[0])
                        
                        # 处理关键点数据
                        kpts = None
                        if keypoints is not None and i < len(keypoints):
                            try:
                                # 获取坐标和置信度并合并
                                kpts_xy = keypoints[i].xy.cpu().numpy()[0]  # [N, 2]
                                kpts_conf = keypoints[i].conf.cpu().numpy()[0]  # [N, 1]
                                kpts = np.concatenate([kpts_xy, kpts_conf[:, None]], axis=1)  # [N, 3]
                            except Exception as e:
                                print(f"处理关键点时出错: {e}")
                                kpts = None
                        
                        # 传递conf_threshold参数
                        if is_box_in_region([x1, y1, x2, y2], kpts, region_points, conf_threshold):
                            # 计算面积
                            area = (x2 - x1) * (y2 - y1)
                            
                            detection = {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": float(conf),
                                "class_id": int(cls_id),
                                "area": float(area),
                                "keypoints": []
                            }
                            
                            # 处理关键点
                            if kpts is not None:
                                for kid in range(len(kpts)):
                                    try:
                                        kpt = kpts[kid]
                                        x, y, kpt_conf = kpt[0], kpt[1], kpt[2]
                                        detection["keypoints"].append({
                                            "name": keypoint_names[kid] if kid < len(keypoint_names) else f"kpt_{kid}",
                                            "x": float(x),
                                            "y": float(y),
                                            "confidence": float(kpt_conf)
                                        })
                                    except Exception as e:
                                        print(f"处理关键点 {kid} 时出错: {e}")
                            
                            frame_data["detections"].append(detection)
                
                data["motions_info"].append(frame_data)
                frame_idx += 1
                pbar.update(1)
    
    finally:
        cap.release()
        print(f"\n处理完成! 共处理 {frame_idx} 帧")
        
        # 保存最终结果
        output_json_path = os.path.join(output_dir, 'final_output.json')
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"结果保存至: {output_json_path}")
    
    return data

def main():
    # 示例用法
    region = [(103, 754), (996, 619), (1771, 727), (1709, 1078), (62, 1078), (102, 753)]
    video_path = "/ssd1/zq/ultralytics-8.3.105/video-test/1.MP4"
    model_path = "/ssd1/zq/ultralytics-8.3.105/runs/pose/train13/weights/best.pt"
    
    data = process_video(
        region_points=region,
        video_path=video_path,
        model_path=model_path,
        output_dir="video_output",
        conf_threshold=0.5,
        device="cuda:0"
    )
    
    print("\n最终数据结构示例:")
    print(json.dumps(data["motions_info"][0], indent=2) if len(data["motions_info"]) > 0 else "无数据")

if __name__ == "__main__":
    main()
