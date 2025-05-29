# import cv2  # 导入 OpenCV 库进行图像处理
# import numpy as np  # 导入 numpy 库进行数值操作
# from ultralytics import YOLO  # 从 ultralytics 包中导入 YOLO 模型
 
# # 将 HSV 颜色转换为 BGR 颜色的函数
# def hsv2bgr(h, s, v):
#     h_i = int(h * 6)  # 将色调转换为整数值
#     f = h * 6 - h_i  # 色调的小数部分
#     p = v * (1 - s)  # 计算不同情况下的值
#     q = v * (1 - f * s)
#     t = v * (1 - (1 - f) * s)
    
#     r, g, b = 0, 0, 0  # 将 RGB 值初始化为 0
 
#     # 根据色调值确定 RGB 值
#     if h_i == 0:
#         r, g, b = v, t, p
#     elif h_i == 1:
#         r, g, b = q, v, p
#     elif h_i == 2:
#         r, g, b = p, v, t
#     elif h_i == 3:
#         r, g, b = p, q, v
#     elif h_i == 4:
#         r, g, b = t, p, v
#     elif h_i == 5:
#         r, g, b = v, p, q
 
#     return int(b * 255), int(g * 255), int(r * 255)  # 返回缩放到 255 的 BGR 值
 
# # 根据 ID 生成随机颜色的函数
# def random_color(id):
#     # 根据 ID 生成色调和饱和度值
#     h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
#     s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
#     return hsv2bgr(h_plane, s_plane, 1)  # 基于 HSV 值返回 BGR 颜色
 
# # 定义关键点之间的骨骼连接
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
#             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
 
# # 定义关键点和肢体的调色板
# pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                          [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                          [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                          [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
 
# # 基于调色板为关键点和肢体分配颜色
# kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
 
# # 主函数
# if __name__ == "__main__":
 
#     # 加载 YOLO 模型
#     model = YOLO("yolov8s-pose.pt")
 
#     # 读取输入图像
#     #img = cv2.imread("ultralytics/assets/bus.jpg")
#     img = cv2.imread("frame_00000.jpg")
#     # 使用 YOLO 进行目标检测
#     results = model(img)[0]
#     names   = results.names  # 获取类别名称
#     boxes   = results.boxes.data.tolist()  # 获取边界框
 
#     # 获取模型检测到的关键点
#     keypoints = results.keypoints.cpu().numpy()
 
#     # 为每个检测到的人绘制关键点和肢体
#     for keypoint in keypoints.data:
#         for i, (x, y, conf) in enumerate(keypoint):
#             color_k = [int(x) for x in kpt_color[i]]  # 获取关键点的颜色
#             if conf < 0.5:
#                 continue
#             if x != 0 and y != 0:
#                 cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)  # 绘制关键点
#         for i, sk in enumerate(skeleton):
#             pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))  # 获取肢体的第一个关键点的位置
#             pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))  # 获取肢体的第二个关键点的位置
 
#             conf1 = keypoint[(sk[0] - 1), 2]  # 第一个关键点的置信度
#             conf2 = keypoint[(sk[1] - 1), 2]  # 第二个关键点的置信度
#             if conf1 < 0.5 or conf2 < 0.5:
#                 continue
#             if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
#                 continue
#             cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)  # 绘制肢体
 
#     # 绘制检测到的对象的边界框和标签
#     for obj in boxes:
#         left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])  # 提取边界框坐标
#         confidence = obj[4]  # 置信度分数
#         label = int(obj[5])  # 类别标签
#         color = random_color(label)  # 为边界框获取随机颜色
#         cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)  # 绘制边界框
#         caption = f"{names[label]} {confidence:.2f}"  # 生成包含类名和置信度分数的标签
#         w, h = cv2.getTextSize(caption, 0, 1, 2)[0]  # 获取文本大小
#         cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)  # 绘制标签背景的矩形
#         cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)  # 放置标签文本
 
#     # 保存标注后的图像
#     cv2.imwrite("predict-pose.jpg", img)
#     print("保存完成")  # 打印保存操作完成的消息

##############简单预测
# from ultralytics import YOLO
# import cv2

# # Load a model
# #model = YOLO('yolov8n-pose.pt')  # load an official model
# model = YOLO('runs/pose/train84/weights/best.pt')  # load a custom model

# # Predict with the model
# results = model(r"C:/Users/Administrator/Desktop/ceshi2.png")  # predict on an image
# # 取第一张（这里只有一张）
# img_with_preds = results[0].plot()  # 返回一张已绘制好框和关键点的 BGR ndarray

# # 用 OpenCV 显示
# cv2.imshow("Pose Predictions", img_with_preds)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

######选取照片进行预测
# from ultralytics import YOLO

# # 加载你自定义训练好的模型
# model = YOLO('yolov8x-pose-p6.pt')

# # 指定图片路径（使用原始字符串）
# image_path = r"C:/Users/Administrator/Desktop/ceshi.png"

# # 模型推理
# results = model.predict(source=image_path)

# # 显示结果（也可以根据需要保存或进一步处理）
# # 假设 results 列表只有一个元素
# results[0].show()

# # from ultralytics import YOLO
# # model = YOLO('runs/pose/train20/weights/best.pt')
# # # 降低 conf-thres、iou-thres
# # results = model.predict(source=r"C:/Users/Administrator/Desktop/ceshi1.jpeg", conf=0.1, iou=0.3, show=True)
# # results[0].show()



# #解决多进程问题
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# def main():
#     # 原有的predict.py代码放在这里
#     model = YOLO('runs/pose/train20/weights/best.pt')  # 改为你的模型路径
    
#     # 降低 conf-thres、iou-thres
#     results = model.predict(source=r"C:/Users/Administrator/Desktop/ceshi1.jpeg", conf=0.1, iou=0.3, show=True)
#     results[0].show()

# if __name__ == '__main__':
#     mp.freeze_support()  # 解决Windows多进程问题
#     main()

####两阶段中的预测模型

from ultralytics import YOLO
import cv2
import numpy as np
import torch.multiprocessing as mp

def process_image(image_path):
    # # 1. 使用原始YOLOv8进行人物检测
    detector = YOLO('yolo11s.pt')  # 纯检测模型
    detections = detector(image_path, classes=0)  # 只检测人类(class 0)
    
    # 2. 使用你训练的姿态估计模型处理每个检测到的人
    pose_model = YOLO('runs/pose/train91/weights/best.pt')
    
    # 读取原始图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, []
        
    result_img = img.copy()
    
    all_keypoints = []
    
    print(f"原始图像检测到 {len(detections[0].boxes)} 个人")
    
    # 对每个检测到的人进行姿态估计
    for det in detections[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, det[:4])
        
        # 扩大边界框以包含完整的人体
        h, w = y2-y1, x2-x1
        x1 = max(0, x1 - int(w*0.1))
        y1 = max(0, y1 - int(h*0.1))
        x2 = min(img.shape[1], x2 + int(w*0.1))
        y2 = min(img.shape[0], y2 + int(h*0.1))
        
        # 裁剪人物区域
        person_img = img[y1:y2, x1:x2]
        if person_img.size == 0:
            print(f"跳过空的裁剪区域: {x1},{y1},{x2},{y2}")
            continue
            
        # 在原图上绘制边界框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 进行姿态估计
        pose_results = pose_model(person_img)
        
        if len(pose_results[0].keypoints) > 0 and pose_results[0].keypoints.shape[1] > 0:
            # 获取关键点
            kpts = pose_results[0].keypoints.data[0].cpu().numpy()
            
            # 检查关键点数组是否为空
            if kpts.size == 0:
                print("关键点数组为空，跳过")
                continue
                
            print(f"获取到关键点形状: {kpts.shape}")
            
            # 将关键点坐标映射回原图
            for i in range(len(kpts)):
                if kpts[i][2] > 0:  # 如果关键点可见
                    kpts[i][0] += x1
                    kpts[i][1] += y1
            
            all_keypoints.append(kpts)
            
            # 在原图上绘制关键点 - 使用不同颜色区分不同部位
            for i, kpt in enumerate(kpts):
                if kpt[2] > 0.5:  # 可见性阈值
                    # 根据关键点类型选择颜色
                    if i <= 4:  # 面部关键点 (鼻子、眼睛、耳朵)
                        color = (255, 0, 0)  # 蓝色
                    elif 5 <= i <= 10:  # 上半身关键点 (肩膀、肘部、手腕)
                        color = (0, 255, 0)  # 绿色
                    else:  # 下半身关键点 (臀部、膝盖、脚踝)
                        color = (0, 255, 255)  # 黄色
                    
                    cv2.circle(result_img, (int(kpt[0]), int(kpt[1])), 5, color, -1)
            
            # 定义COCO格式的17个关键点的骨架连接
            num_kpts = len(kpts)
            if num_kpts == 17:  # COCO格式
                # 面部连接 - 蓝色
                face_skeleton = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4]]
                
                # 上半身连接 - 绿色
                upper_body_skeleton = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12]]
                
                # 下半身连接 - 红色
                lower_body_skeleton = [[11, 13], [13, 15], [12, 14], [14, 16]]
                
                # 额外的连接 - 紫色
                extra_connections = [[3, 5], [4, 6]]
                
                # 所有骨架连接
                skeleton_parts = {
                    "face": {"connections": face_skeleton, "color": (255, 0, 0)},  # 蓝色
                    "upper_body": {"connections": upper_body_skeleton, "color": (0, 255, 0)},  # 绿色
                    "lower_body": {"connections": lower_body_skeleton, "color": (0, 0, 255)},  # 红色
                    "extra": {"connections": extra_connections, "color": (255, 0, 255)}  # 紫色
                }
                
                # 绘制所有骨架部分
                for part_name, part_info in skeleton_parts.items():
                    connections = part_info["connections"]
                    color = part_info["color"]
                    
                    for sk in connections:
                        if max(sk) >= num_kpts:
                            print(f"警告: 骨架索引 {sk} 超出关键点范围 {num_kpts}")
                            continue
                            
                        if kpts[sk[0]][2] > 0.5 and kpts[sk[1]][2] > 0.5:
                            pt1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
                            pt2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))
                            cv2.line(result_img, pt1, pt2, color, 2)
            else:
                # 根据你的关键点数量自定义骨架
                print(f"检测到未知关键点格式，数量为 {num_kpts}，跳过骨架绘制")
        else:
            print(f"未检测到关键点")
    
    # 保存结果
    cv2.imwrite('multi_person_pose_result.jpg', result_img)
    print(f"检测到 {len(all_keypoints)} 个人的姿态")
    
    return result_img, all_keypoints

def main():
    try:
        result_img, keypoints = process_image(r"C:/Users/Administrator/Desktop/ceshi1.png")
        if result_img is not None:
            cv2.imwrite('output_result.jpg', result_img)
            print(f"结果已保存至 output_result.jpg")
    except Exception as e:
        import traceback
        print(f"处理过程中出错: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    mp.freeze_support()
    main()


# #####对前20个epoch进行一个预测

# from ultralytics import YOLO
# import cv2
# import numpy as np
# import torch
# from pathlib import Path

# def multi_epoch_pose(image_path, 
#                      detect_model_path='yolo11s.pt',
#                      pose_base_path='yolov8s-pose.pt',
#                      pose_epochs_dir='training_visualizations',
#                      output_dir='multi_epoch_results',
#                      vis_kpt_threshold=0.5):
#     """
#     对同一张图片：
#       1. 用检测模型检测人物；
#       2. 对每个检测到的人，循环加载每个 epoch 的 state_dict（保存在 pose_epochs_dir）进行姿态预测；
#       3. 绘制关键点和骨架连线（COCO 17 点），并保存每个 epoch 的结果在 output_dir/epoch_{i}/predict.jpg
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True)
    
#     # 1. 加载检测模型
#     detector = YOLO(detect_model_path)
    
#     # 2. 读取原图
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图像: {image_path}")
    
#     # 3. 使用检测模型检测人物（只检测 class 0: 人）
#     det_result = detector(image_path, classes=0)
#     boxes = det_result[0].boxes.xyxy.cpu().numpy()  # 每一行 [x1, y1, x2, y2]
#     print(f"检测到 {len(boxes)} 个人")
    
#     # 4. 遍历每个 epoch 的保存目录
#     epoch_dirs = sorted(Path(pose_epochs_dir).glob("epoch_*"))
#     for epoch_dir in epoch_dirs:
#         epoch = epoch_dir.name.split('_')[-1]
#         weight_file = epoch_dir / f"model_{epoch}.pt"
#         if not weight_file.exists():
#             print(f"[跳过] 未找到 {weight_file}")
#             continue

#         # 4.1 新建输出子目录
#         out_sub = output_dir / f"epoch_{epoch}"
#         out_sub.mkdir(exist_ok=True)
        
#         # 4.2 构造基础姿态模型（用于获得模型结构）
#         base_pose = YOLO(pose_base_path)  # 用作基础架构，需与训练时保持一致
#         try:
#             # 4.3 手动加载 state_dict（你保存的文件只有 state_dict）
#             state_dict = torch.load(str(weight_file), map_location='cpu')
#             base_pose.model.model.load_state_dict(state_dict)
#         except Exception as e:
#             print(f"加载权重 {weight_file} 时出错: {e}")
#             continue
        
#         # 4.4 对每个检测到的人进行姿态预测
#         # 为当前 epoch 新建一份可视化图（复制原图）
#         vis = img.copy()
        
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box[:4])
#             # 扩大边界框以确保完整包含人体
#             w, h = x2 - x1, y2 - y1
#             x1e = max(0, int(x1 - 0.1 * w))
#             y1e = max(0, int(y1 - 0.1 * h))
#             x2e = min(img.shape[1], int(x2 + 0.1 * w))
#             y2e = min(img.shape[0], int(y2 + 0.1 * h))
#             crop = img[y1e:y2e, x1e:x2e]
#             if crop.size == 0:
#                 continue

#             # 4.5 使用当前 epoch 的模型对裁剪图做姿态预测
#             preds = base_pose(crop)
#             # 这里假定每个裁剪图只含一人，取第一个检测结果
#             if preds and preds[0].keypoints is not None and preds[0].keypoints.shape[1] > 0:
#                 # 获取关键点，shape 通常为 (num_kpts, 3)
#                 kpts = preds[0].keypoints.data[0].cpu().numpy()
#                 if kpts.size == 0:
#                     continue

#                 # 将关键点坐标映射回原图（加上裁剪区域的偏移量）
#                 for j in range(len(kpts)):
#                     if kpts[j][2] > vis_kpt_threshold:
#                         kpts[j][0] += x1e
#                         kpts[j][1] += y1e
#                         # 绘制关键点，绿色圆点
#                         cv2.circle(vis, (int(kpts[j][0]), int(kpts[j][1])), 3, (0, 255, 0), -1)

#                 # 绘制骨架连线，如果关键点数为 17（COCO 格式）
#                 num_kpts = len(kpts)
#                 if num_kpts == 17:
#                     # 定义各部分骨架连接（以不同颜色绘制）
#                     face_skeleton = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4]]
#                     upper_body_skeleton = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12]]
#                     lower_body_skeleton = [[11, 13], [13, 15], [12, 14], [14, 16]]
#                     extra_connections = [[3, 5], [4, 6]]
                    
#                     skeleton_parts = {
#                         "face": {"connections": face_skeleton, "color": (255, 0, 0)},      # 蓝色
#                         "upper_body": {"connections": upper_body_skeleton, "color": (0, 255, 0)},  # 绿色
#                         "lower_body": {"connections": lower_body_skeleton, "color": (0, 0, 255)},  # 红色
#                         "extra": {"connections": extra_connections, "color": (255, 0, 255)}   # 紫色
#                     }
                    
#                     for part_info in skeleton_parts.values():
#                         for conn in part_info["connections"]:
#                             idx1, idx2 = conn
#                             # 仅当两个关键点的置信度均高于阈值时绘制连线
#                             if kpts[idx1][2] > vis_kpt_threshold and kpts[idx2][2] > vis_kpt_threshold:
#                                 pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
#                                 pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
#                                 cv2.line(vis, pt1, pt2, part_info["color"], 2)
#             else:
#                 print("未检测到关键点或预测为空！")
        
#         # 4.6 保存当前 epoch 的预测可视化结果
#         save_path = out_sub / "predict.jpg"
#         cv2.imwrite(str(save_path), vis)
#         print(f"[epoch {epoch}] 预测可视化结果已保存至 {save_path}")
    
#     print("所有 epoch 的预测已完成。")



# if __name__ == "__main__":
#     image_path = r"C:/Users/Administrator/Desktop/ceshi.png"
#     multi_epoch_pose(image_path)
