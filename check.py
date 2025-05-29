# from ultralytics import YOLO

# # 加载模型（确保加载的是 pose 版本）
# model = YOLO('yolov8s-pose.pt')

# # 输出所有参数的名称
# for name, param in model.model.named_parameters():
#     print(name)
# 假设 pose head 模块的类型为 Pose
# for name, module in model.model.named_modules():
#     if module.__class__.__name__ == "Pose":
#         print("找到 Pose head 模块：", name)


# from ultralytics import YOLO

# # 加载预训练的 pose 模型
# model = YOLO('yolov8s-pose.pt')
# # 打印模型结构
# print(model.model)

# from ultralytics import YOLO

# # 加载官方模型
# official_model = YOLO('yolov8s-pose.pt')

# # 加载你训练后保存的模型
# trained_model = YOLO('runs/pose/train11/weights/best.pt')

# # 初始化一个列表来记录参数发生变化的层
# changed_layers = []

# # 遍历模型的每一层
# for (name1, param1), (name2, param2) in zip(official_model.model.named_parameters(), trained_model.model.named_parameters()):
#     # 比较参数是否相同
#     if not (param1 == param2).all():
#         changed_layers.append(name1)

# # 打印参数发生变化的层
# print("参数发生变化的层：")
# for layer in changed_layers:
#     print(layer)


# from ultralytics import YOLO

# # 加载模型
# model = YOLO('yolov8n-pose.pt')

# # 查看 Pose 模块（pose-head）
# pose_head_layer = model.model.model[22]
# print("Pose Head 层:")
# print(pose_head_layer)

# # 查看相关依赖层
# print("\n相关依赖层:")
# for layer_index in [15, 18, 21]:
#     layer = model.model.model[layer_index]
#     print(f"层 {layer_index}:")
#     print(layer)


# from ultralytics import YOLO
# model = YOLO('yolov8s-pose.pt')
# # 查看顶层分段
# print(model.model)  

# # 查看 DetectPose 头里的所有子模块
# head = model.model.model[-1]
# for name, module in head.named_modules():
#     print(name)



# import onnx

# # 尝试直接 load & checker
# model = onnx.load("C:/Users/Administrator/Desktop/X-AnyLabeling/yolo11x-pose.onnx")
# onnx.checker.check_model(model)
# print("✔️ ONNX model is valid!")


# import os

# def count_images_in_folder(folder_path):
#     # 图片扩展名列表，添加你所需要支持的格式
#     image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
#     image_count = 0
    
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if any(file.lower().endswith(ext) for ext in image_extensions):
#                 image_count += 1
                
#     return image_count

# def main():
#     base_folder = r"C:/Users/Administrator/Desktop/标注数据集"  # 请替换为你的文件夹路径
    
#     left_folder = os.path.join(base_folder, 'left')
#     right_folder = os.path.join(base_folder, 'right')
#     back_folder = os.path.join(base_folder, 'back')
    
#     total_images = 0
    
#     total_images += count_images_in_folder(left_folder)
#     total_images += count_images_in_folder(right_folder)
#     total_images += count_images_in_folder(back_folder)
    
#     print(f"Total number of images: {total_images}")

# if __name__ == '__main__':
#     main()

# import json
# from pathlib import Path

# # 直接在这里指定你的 JSON 文件路径
# JSON_PATH = Path("Z:/BadData/Benchmark/Benchmark_pose/rawdata/0522龙猫数据-人体关键点及拉框-720/0522龙猫数据-人体关键点及拉框-720/images/标注数据集/left/2/frame_00033.json")

# def replace_kie_linking(obj):
#     if isinstance(obj, dict):
#         if obj.get("kie_linking", None) is False:
#             obj["kie_linking"] = []
#         for key, val in obj.items():
#             obj[key] = replace_kie_linking(val)
#         return obj
#     elif isinstance(obj, list):
#         return [replace_kie_linking(item) for item in obj]
#     else:
#         return obj

# def main():
#     # 读取原文件
#     data = json.loads(JSON_PATH.read_text(encoding='utf-8'))
#     # 替换
#     updated = replace_kie_linking(data)
#     # 覆盖写回
#     JSON_PATH.write_text(json.dumps(updated, ensure_ascii=False, indent=4), encoding='utf-8')
#     print(f"已在原文件 {JSON_PATH} 上完成替换。")

# if __name__ == "__main__":
#     main()


##############可视化图片+txt文件########################
# import cv2
# import os

# # --- 硬编码文件夹路径 ---
# # TXT标注文件夹路径
# labels_dir = r"Z:/BadData\Benchmark/Benchmark_pose/labels"
# # 图片文件夹路径
# images_dir = r"Z:/BadData/Benchmark/Benchmark_pose/images"

# # --- 可视化参数 ---
# # 目标框颜色 (BGR格式: Blue, Green, Red)
# # 可以为不同的class_id设置不同颜色，这里为1和2都使用绿色
# bbox_color = (0, 255, 0)  # 绿色
# # 关键点颜色 (BGR格式)
# kp_color = (0, 0, 255)    # 红色
# # 关键点连接线颜色 (骨架) (BGR格式)
# line_color = (255, 0, 0)  # 蓝色
# # 关键点半径 (像素)
# kp_radius = 5
# # 线条粗细 (像素)
# thickness = 2

# # 标准COCO 17关键点的连接关系 (索引从0开始)
# # 用于绘制骨架
# # 0:鼻尖, 1:左眼, 2:右眼, 3:左耳, 4:右耳,
# # 5:左肩, 6:右肩, 7:左手肘, 8:右手肘,
# # 9:左手腕, 10:右手腕, 11:左髋, 12:右髋,
# # 13:左膝, 14:右膝, 15:左脚踝, 16:右脚踝
# standard_skeleton = [
#     (0, 1), (0, 2), (1, 3), (2, 4), # Head
#     (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), # Arms
#     (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), # Legs
#     (5, 11), (6, 12) # Shoulders to Hips
# ]

# # --- 遍历TXT文件并可视化 ---
# txt_files_found = 0
# processed_count = 0

# # 检查文件夹是否存在
# if not os.path.isdir(labels_dir):
#     print(f"错误：标签文件夹未找到：{labels_dir}")
#     exit()
# if not os.path.isdir(images_dir):
#     print(f"错误：图片文件夹未找到：{images_dir}")
#     exit()

# print(f"开始遍历标签文件夹：{labels_dir}")

# # 遍历 labels_dir 及其所有子文件夹
# for root, _, files in os.walk(labels_dir):
#     for file in files:
#         # 仅处理以 .txt 结尾的文件
#         if file.endswith(".txt"):
#             txt_files_found += 1
#             txt_path = os.path.join(root, file)

#             # 构建对应的图片文件路径
#             # 获取从 labels_dir 到当前 txt 文件的相对路径
#             relative_path = os.path.relpath(txt_path, labels_dir)
#             # 将相对路径中的 .txt 扩展名替换为 .jpg (假设图片是jpg格式)
#             # 你可能需要根据你的实际图片格式修改这里的 ".jpg"
#             image_relative_path = relative_path.replace(".txt", ".jpg")
#             # 构建完整的图片路径
#             image_path = os.path.join(images_dir, image_relative_path)

#             print(f"\n--- 处理文件: {relative_path} ---")

#             # 检查对应的图片文件是否存在
#             if not os.path.exists(image_path):
#                 print(f"警告：对应的图片文件未找到：{image_path}，跳过此标注文件。")
#                 continue

#             # --- 读取图片 ---
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"警告：无法读取图片文件：{image_path}，跳过此标注文件。")
#                 continue

#             img_h, img_w, _ = img.shape
#             print(f"成功读取图片：{image_path} (尺寸: {img_w}x{img_h})")

#             # --- 读取并解析TXT文件 ---
#             annotations = []
#             try:
#                 with open(txt_path, 'r') as f:
#                     for line_num, line in enumerate(f):
#                         parts = line.strip().split()
#                          # 检查行的结构是否符合预期: class_id + 4 bbox + 17 * 3 keypoints = 56
#                         if len(parts) != 1 + 4 + 17 * 3:
#                              print(f"警告 (文件 {relative_path}, 行 {line_num+1}): 跳过结构不完整的行 ({len(parts)} 部分): {line.strip()}")
#                              continue

#                         try:
#                             class_id = int(parts[0])
#                             # 只处理 class_id 为 1 或 2 的目标
#                             if class_id in [1, 2]:
#                                 # 解析目标框 (中心点x, 中心点y, 宽度, 高度) - 归一化坐标
#                                 cx, cy, w, h = map(float, parts[1:5])

#                                 # 解析关键点 (x, y, 可见度) - 归一化坐标
#                                 keypoints_data = list(map(float, parts[5:]))

#                                 annotations.append({
#                                     'class_id': class_id,
#                                     'bbox': (cx, cy, w, h),
#                                     'keypoints': keypoints_data
#                                 })
#                         except ValueError as e:
#                              print(f"警告 (文件 {relative_path}, 行 {line_num+1}): 跳过包含无效数值的行 (解析错误: {e}): {line.strip()}")
#                         except Exception as e:
#                              print(f"警告 (文件 {relative_path}, 行 {line_num+1}): 跳过因未知错误无法解析的行 ({e}): {line.strip()}")

#                 print(f"从 {relative_path} 共解析到 {len(annotations)} 个 class_id 为 1 或 2 的目标.")

#                 # --- 在图片上绘制目标框和关键点 ---
#                 # 复制一份图片用于绘制，保留原图
#                 img_annotated = img.copy()

#                 if len(annotations) > 0: # 只有当解析到目标时才进行绘制
#                     for ann in annotations:
#                         # 绘制目标框
#                         cx, cy, w, h = ann['bbox']
#                         # 转换归一化坐标到像素坐标 (确保是整数)
#                         x_min = int((cx - w/2) * img_w)
#                         y_min = int((cy - h/2) * img_h)
#                         x_max = int((cx + w/2) * img_w)
#                         y_max = int((cy + h/2) * img_h)
#                         # 确保坐标在图片范围内
#                         x_min = max(0, x_min)
#                         y_min = max(0, y_min)
#                         x_max = min(img_w - 1, x_max)
#                         y_max = min(img_h - 1, y_max)

#                         cv2.rectangle(img_annotated, (x_min, y_min), (x_max, y_max), bbox_color, thickness)

#                         # 绘制关键点和连接线
#                         keypoints_data = ann['keypoints']
#                         visible_kps_pixel = {} # 存储可见关键点的像素坐标 {索引: (x, y)}

#                         for i in range(17):
#                             # 检查是否有足够的关键点数据
#                             if (i * 3 + 2) >= len(keypoints_data):
#                                 # print(f"警告: 文件 {relative_path} 中的关键点数据不足 (目标 {annotations.index(ann)+1}, 关键点 {i+1})")
#                                 break # 跳出关键点绘制循环

#                             kp_x, kp_y, kp_v = keypoints_data[i*3], keypoints_data[i*3+1], keypoints_data[i*3+2]

#                             # 仅处理可见度 >= 1.0 (可见或被遮挡) 的关键点
#                             if kp_v >= 1.0:
#                                 # 转换归一化坐标到像素坐标 (确保是整数)
#                                 kp_x_abs = int(kp_x * img_w)
#                                 kp_y_abs = int(kp_y * img_h)

#                                 # 确保坐标在图片范围内
#                                 kp_x_abs = max(0, min(img_w - 1, kp_x_abs))
#                                 kp_y_abs = max(0, min(img_h - 1, kp_y_abs))

#                                 # 绘制关键点
#                                 cv2.circle(img_annotated, (kp_x_abs, kp_y_abs), kp_radius, kp_color, -1) # -1 表示填充圆
#                                 visible_kps_pixel[i] = (kp_x_abs, kp_y_abs) # 记录可见关键点的像素坐标

#                         # 绘制关键点连接线 (骨架)
#                         for kp1_idx, kp2_idx in standard_skeleton:
#                             # 只有当连接的两个关键点都在 visible_kps_pixel 中时才绘制连接线
#                             if kp1_idx in visible_kps_pixel and kp2_idx in visible_kps_pixel:
#                                 pt1 = visible_kps_pixel[kp1_idx]
#                                 pt2 = visible_kps_pixel[kp2_idx]
#                                 cv2.line(img_annotated, pt1, pt2, line_color, thickness)

#                 processed_count += 1
#                 # --- 显示图片 ---
#                 # 可以使用 cv2.WINDOW_NORMAL 来调整窗口大小
#                 cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)
#                 cv2.imshow("Visualization", img_annotated)

#                 print(f"已可视化 {processed_count}/{txt_files_found} 个文件。按下任意键继续，按下 'q' 键退出。")

#                 # 等待按键，按下 'q' 键退出循环
#                 key = cv2.waitKey(0)
#                 if key == ord('q'):
#                     print("用户请求退出。")
#                     break # 跳出文件循环

#             except FileNotFoundError:
#                 print(f"警告：无法打开标注文件：{txt_path}，跳过。")
#             except Exception as e:
#                 print(f"处理文件 {relative_path} 时发生未知错误：{e}，跳过。")

#     # 如果用户按下了 'q'，则在外层循环也跳出
#     if 'key' in locals() and key == ord('q'):
#         break # 跳出 os.walk 循环


# # --- 清理和结束 ---
# cv2.destroyAllWindows()
# print("\n可视化完成。")
# print(f"总共找到 {txt_files_found} 个 .txt 文件。")
# print(f"成功处理并可视化 {processed_count} 个文件。")


import os
import glob

def check_id_count(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    id1_count = 0
    id2_count = 0
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        try:
            class_id = int(parts[0])
            if class_id == 1:
                id1_count += 1
            elif class_id == 2:
                id2_count += 1
        except ValueError:
            print(f"Warning: Invalid class ID in file {file_path}")
    
    total = id1_count + id2_count
    return file_path, id1_count, id2_count, total

def main(directory):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    
    if not txt_files:
        print(f"No txt files found in {directory}")
        return
    
    issues = []
    
    for file_path in txt_files:
        file_name, id1, id2, total = check_id_count(file_path)
        if total != 2:
            issues.append((file_name, id1, id2, total))
    
    if issues:
        print(f"Found {len(issues)} files with id1+id2 count != 2:")
        for file_name, id1, id2, total in issues:
            print(f"{os.path.basename(file_name)}: id1={id1}, id2={id2}, total={total}")
    else:
        print(f"All {len(txt_files)} files have exactly 2 instances of id1+id2.")

if __name__ == "__main__":
    # 替换为你的txt文件夹路径
    directory = r"Z:/BadData\Benchmark/Benchmark_pose/labels/1/right/23"
    main(directory)
