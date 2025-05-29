# #第一近
# import os
# import json
# import cv2
# import numpy as np

# # 路径配置
# image_folder = r"Z:\Badminton_HAR_O\train\CLEAN\0227_lindan09_left_clip1_0-150_images"
# json_folder = r"Z:\Badminton_HAR_O\train\CLEAN\0227_lindan09_left_clip1_0-150_union1"
# output_folder = r"Z:\Badminton_HAR_O\train\CLEAN\0227_lindan09_left_clip1_0-150_1"


# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# def calculate_bbox_area(points):
#     """
#     计算任意顺序的矩形多边形的面积：
#     根据所有坐标的最小值和最大值计算出包围框面积。
#     """
#     if not points or len(points) < 2:
#         return 0
#     xs = [p[0] for p in points]
#     ys = [p[1] for p in points]
#     width = max(xs) - min(xs)
#     height = max(ys) - min(ys)
#     return abs(width * height)

# def select_largest_person_group_id(json_data):
#     """
#     遍历 json_data 中所有 label 为 "person" 且 shape_type 为 "rectangle" 的标注，
#     根据目标框面积选出最大的目标，并返回其 group_id。
#     """
#     largest_area = 0
#     largest_group_id = None

#     for shape in json_data.get("shapes", []):
#         if shape.get("label") == "person" and shape.get("shape_type") == "rectangle":
#             points = shape.get("points", [])
#             area = calculate_bbox_area(points)
#             if area > largest_area:
#                 largest_area = area
#                 largest_group_id = shape.get("group_id")
    
#     return largest_group_id

# def filter_json_by_group_id(json_data, group_id):
#     """
#     根据指定的 group_id，从 json_data 中筛选出所有对应的标注，
#     同时保留其它关键信息，构造新的 JSON 数据。
#     """
#     new_json = {
#         "version": json_data.get("version", "2.5.4"),
#         "flags": json_data.get("flags", {}),
#         "shapes": [],
#         "imagePath": json_data.get("imagePath", ""),
#         "imageData": json_data.get("imageData", None),
#         "imageHeight": json_data.get("imageHeight", None),
#         "imageWidth": json_data.get("imageWidth", None),
#         "description": json_data.get("description", "")
#     }
    
#     for shape in json_data.get("shapes", []):
#         if shape.get("group_id") == group_id:
#             new_json["shapes"].append(shape)
            
#     return new_json

# def process_json_files():
#     """
#     遍历 image_folder 中的图片文件，
#     读取对应 JSON 文件后选出面积最大的“person”标注的 group_id，
#     将该 group_id 对应的所有标注保存到新的 JSON 文件中。
#     """
#     image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     print(f"找到 {len(image_files)} 个图片文件。")
    
#     for image_file in image_files:
#         image_path = os.path.join(image_folder, image_file)
#         # JSON 文件名默认为与图片同名（扩展名为 .json）
#         json_filename = os.path.splitext(image_file)[0] + ".json"
#         json_path = os.path.join(json_folder, json_filename)
        
#         if not os.path.exists(json_path):
#             print(f"警告：图片 {image_file} 对应的 JSON 文件不存在！")
#             continue

#         # 读取 JSON 数据
#         with open(json_path, "r", encoding="utf-8") as f:
#             json_data = json.load(f)
        
#         # 选出面积最大的目标的 group_id
#         largest_group_id = select_largest_person_group_id(json_data)
#         if largest_group_id is None:
#             print(f"警告：在 {json_filename} 中未找到有效的 person 标注！")
#             continue
        
#         print(f"在 {json_filename} 中，选出的最大 person group_id 是：{largest_group_id}")
        
#         # 根据选定的 group_id 筛选对应标注并构建新的 JSON 数据
#         new_json_data = filter_json_by_group_id(json_data, largest_group_id)
        
#         # 保存新的 JSON 文件到 output_folder 中，文件名与原 JSON 文件一致
#         output_json_path = os.path.join(output_folder, json_filename)
#         with open(output_json_path, "w", encoding="utf-8") as f:
#             json.dump(new_json_data, f, indent=4, ensure_ascii=False)
#         print(f"已保存 {json_filename} 新 JSON 文件到 {output_json_path}")

# if __name__ == "__main__":
#     process_json_files()



#第二近
# import os
# import json
# import cv2
# import numpy as np

# # 路径配置，根据实际情况修改
# image_folder = r"Z:\Badminton_HAR_O\train\CLEAN\0227_lindan09_left_clip1_0-150_images"
# json_folder = r"Z:\Badminton_HAR_O\train\CLEAN\0227_lindan09_left_clip1_0-150_union1"
# output_folder = r"Z:\Badminton_HAR_O\train\CLEAN\0227_lindan09_left_clip1_0-150_2"


# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# def calculate_bbox_area(points):
#     """
#     计算任意顺序的矩形多边形的面积：
#     根据所有坐标的最小值和最大值计算出包围框面积。
#     """
#     if not points or len(points) < 2:
#         return 0
#     xs = [p[0] for p in points]
#     ys = [p[1] for p in points]
#     width = max(xs) - min(xs)
#     height = max(ys) - min(ys)
#     return abs(width * height)

# def select_second_largest_person_group_id(json_data):
#     """
#     遍历 json_data 中所有 label 为 "person" 且 shape_type 为 "rectangle" 的标注，
#     计算每个标注的面积，然后返回面积第二大的目标的 group_id。
#     如果只有一个符合条件的标注，则返回 None。
#     """
#     person_shapes = []
    
#     # 收集所有符合条件的标注及其面积
#     for shape in json_data.get("shapes", []):
#         if shape.get("label") == "person" and shape.get("shape_type") == "rectangle":
#             points = shape.get("points", [])
#             area = calculate_bbox_area(points)
#             person_shapes.append((area, shape.get("group_id")))
    
#     # 如果符合条件的标注少于2个，无法选择第二大，返回 None
#     if len(person_shapes) < 2:
#         return None

#     # 按面积降序排序
#     person_shapes.sort(key=lambda x: x[0], reverse=True)
    
#     # 返回第二大目标的 group_id
#     return person_shapes[1][1]

# def filter_json_by_group_id(json_data, group_id):
#     """
#     根据指定的 group_id，从 json_data 中筛选出所有对应的标注，
#     同时保留其它关键信息，构造新的 JSON 数据。
#     """
#     new_json = {
#         "version": json_data.get("version", "2.5.4"),
#         "flags": json_data.get("flags", {}),
#         "shapes": [],
#         "imagePath": json_data.get("imagePath", ""),
#         "imageData": json_data.get("imageData", None),
#         "imageHeight": json_data.get("imageHeight", None),
#         "imageWidth": json_data.get("imageWidth", None),
#         "description": json_data.get("description", "")
#     }
    
#     for shape in json_data.get("shapes", []):
#         if shape.get("group_id") == group_id:
#             new_json["shapes"].append(shape)
            
#     return new_json

# def process_json_files():
#     """
#     遍历 image_folder 中的图片文件，
#     读取对应 JSON 文件后选出面积第二大的“person”标注的 group_id，
#     将该 group_id 对应的所有标注保存到新的 JSON 文件中。
#     """
#     image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     print(f"找到 {len(image_files)} 个图片文件。")
    
#     for image_file in image_files:
#         image_path = os.path.join(image_folder, image_file)
#         # JSON 文件名默认为与图片同名（扩展名为 .json）
#         json_filename = os.path.splitext(image_file)[0] + ".json"
#         json_path = os.path.join(json_folder, json_filename)
        
#         if not os.path.exists(json_path):
#             print(f"警告：图片 {image_file} 对应的 JSON 文件不存在！")
#             continue

#         # 读取 JSON 数据
#         with open(json_path, "r", encoding="utf-8") as f:
#             json_data = json.load(f)
        
#         # 选出面积第二大的目标的 group_id
#         second_largest_group_id = select_second_largest_person_group_id(json_data)
#         if second_largest_group_id is None:
#             print(f"警告：在 {json_filename} 中未找到第二大的 person 标注！")
#             continue
        
#         print(f"在 {json_filename} 中，选出的第二大 person group_id 是：{second_largest_group_id}")
        
#         # 根据选定的 group_id 筛选对应标注并构建新的 JSON 数据
#         new_json_data = filter_json_by_group_id(json_data, second_largest_group_id)
        
#         # 保存新的 JSON 文件到 output_folder 中，文件名与原 JSON 文件一致
#         output_json_path = os.path.join(output_folder, json_filename)
#         with open(output_json_path, "w", encoding="utf-8") as f:
#             json.dump(new_json_data, f, indent=4, ensure_ascii=False)
#         print(f"已保存 {json_filename} 新 JSON 文件到 {output_json_path}")

# if __name__ == "__main__":
#     process_json_files()


##########################重新安排关键点的顺序########################


# import os
# import json

# # 1. 你要处理的 JSON 文件夹
# input_dir = r"C:/Users/Administrator/Desktop/jianjson"
# # 2. 输出到新的文件夹，避免覆盖原始数据
# output_dir = r"C:/Users/Administrator/Desktop/900/jian"
# os.makedirs(output_dir, exist_ok=True)

# # 关键点的期望顺序
# keypoint_order = [
#     'nose',
#     'left_eye', 'right_eye',
#     'left_ear', 'right_ear',
#     'left_shoulder', 'right_shoulder',
#     'left_elbow', 'right_elbow',
#     'left_wrist', 'right_wrist',
#     'left_hip', 'right_hip',
#     'left_knee', 'right_knee',
#     'left_ankle', 'right_ankle',
# ]

# for fname in os.listdir(input_dir):
#     if not fname.lower().endswith('.json'):
#         continue

#     path_in = os.path.join(input_dir, fname)
#     with open(path_in, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     shapes = data.get('shapes', [])

#     # 按 group_id 分组
#     groups = {}
#     for shape in shapes:
#         gid = shape.get('group_id', None)
#         groups.setdefault(gid, []).append(shape)

#     new_shapes = []
#     # 修改排序: 如果 group_id 为 None，则赋予一个很大的数字，使其排在后面
#     def sort_key(x):
#         if x is None:
#             return float('inf')
#         # 如果能转换成整数，则转换，这里默认组 id 可能是整数或字符串数字
#         if isinstance(x, (int, str)) and str(x).isdigit():
#             return int(x)
#         return x

#     for gid in sorted(groups.keys(), key=sort_key):
#         grp = groups[gid]

#         # 1) 保留原有的目标框（矩形框）
#         rects = [s for s in grp if s['shape_type'] == 'rectangle' and s['label'] == 'person']
#         new_shapes.extend(rects)

#         # 2) 按照关键点顺序添加关键点信息
#         for kp in keypoint_order:
#             pts = [s for s in grp if s['shape_type'] == 'point' and s['label'] == kp]
#             new_shapes.extend(pts)

#         # 3) 保留其他非关键点信息
#         others = [s for s in grp if s not in rects and s['label'] not in keypoint_order]
#         new_shapes.extend(others)

#     # 更新并写入 JSON 文件
#     data['shapes'] = new_shapes
#     path_out = os.path.join(output_dir, fname)
#     with open(path_out, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)

#     print(f"Processed {fname} → {path_out}")

#可视化
# import json
# import os
# import cv2
# import matplotlib.pyplot as plt

# # 设置图片和JSON文件的路径
# image_folder = r"C:/Users/Administrator/Desktop/data/images" # 图片文件夹
# json_folder = r"C:/Users/Administrator/Desktop/data/labels_json"  # JSON文件夹

# def visualize_image_with_json(image_path, json_path):
#     """
#     加载图像和JSON文件，绘制标注的关键点。
#     """
#     # 读取图片
#     image = cv2.imread(image_path)
    
#     # 将 BGR 转换为 RGB，因为 OpenCV 默认是 BGR，而 matplotlib 使用 RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # 读取对应的 JSON 文件
#     with open(json_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)

#     shapes = json_data.get('shapes', [])
#     print(f"Found {len(shapes)} shapes in {json_path}.")
    
#     # 遍历所有 shapes，根据标签（label）绘制关键点
#     for shape in shapes:
#         if shape.get('label') in ['nose', 'left_eye', 'right_eye', 'left_shoulder', 'right_shoulder']:  # 可以根据需要添加更多标签
#             nose_coords = shape.get('points', [None])[0]
#             if nose_coords is not None:
#                 # 在图像上绘制关键点
#                 x, y = int(nose_coords[0]), int(nose_coords[1])
#                 cv2.circle(image_rgb, (x, y), 5, (255, 0, 0), -1)  # 绘制红色圆点
    
#     # 显示图像
#     plt.imshow(image_rgb)
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()

# def process_images_and_json():
#     """
#     遍历所有图像和对应的JSON文件，进行可视化。
#     """
#     # 获取所有图片文件
#     image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     print(f"Found {len(image_files)} image files.")
    
#     for image_file in image_files:
#         image_path = os.path.join(image_folder, image_file)
#         json_file = image_file.replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json')  # 假设 JSON 文件与图片文件同名
#         json_path = os.path.join(json_folder, json_file)
        
#         if os.path.exists(json_path):
#             print(f"Visualizing {image_file} with {json_file}...")
#             visualize_image_with_json(image_path, json_path)
#         else:
#             print(f"Warning: No JSON file found for {image_file}.")

# if __name__ == "__main__":
#     process_images_and_json()

# #选择出不完整的json文件
# import json
# import os

# def check_json_completeness(folder_path):
#     incomplete_files = []

#     # Iterate through each json file
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.json'):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 # Load the JSON data
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)

#                 # Check if there is a "person" target box
#                 has_person_box = False
#                 has_keypoints = False
                
#                 # Iterate through the shapes to check for "person" box and keypoints
#                 for shape in data.get('shapes', []):
#                     if shape.get('label') == 'person' and shape.get('shape_type') == 'rectangle':
#                         has_person_box = True
#                     elif shape.get('label') in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
#                                                 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
#                                                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
#                                                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
#                         has_keypoints = True
                
#                 # Check for missing components
#                 if not has_person_box or not has_keypoints:
#                     incomplete_files.append(filename)

#             except Exception as e:
#                 print(f"Error reading {filename}: {e}")

#     return incomplete_files


# # Example usage:
# folder_path = r"Z:/Badminton_HAR_O/train/CLEAN/0227_lindan09_left_clip1_0-150_2"
# incomplete_files = check_json_completeness(folder_path)

# # Print out the incomplete files
# if incomplete_files:
#     print("Incomplete JSON files detected:")
#     for file in incomplete_files:
#         print(file)
# else:
#     print("All JSON files are complete.")

# #######
# import os
# import json

# # 定义输入和输出目录
# input_dir = r"Z:/Badminton_HAR_O/train/CLEAN/0227_lindan09_left_clip1_0-150_union1"           # 输入 JSON 文件所在目录
# output_dir = r"Z:/Badminton_HAR_O/train/CLEAN/check2json"    # 输出 JSON 文件所在目录

# # 如果输出目录不存在，则创建
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 定义各 JSON 文件对应的目标 group_id（根据实际情况调整）
# # 键为文件名，值为需要保留的 group_id
# file_group_mapping = {
#     "frame_00067.json": 2,

# }

# # 遍历 mapping 中的每个 JSON 文件
# for filename, target_group in file_group_mapping.items():
#     input_path = os.path.join(input_dir, filename)
#     output_path = os.path.join(output_dir, filename)
    
#     if not os.path.exists(input_path):
#         print(f"文件 {input_path} 不存在，跳过。")
#         continue

#     try:
#         # 读取 JSON 数据
#         with open(input_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"读取 {filename} 时出错：{e}")
#         continue

#     # 处理 "shapes" 数组：仅保留 group_id 匹配的项目
#     if "shapes" in data and isinstance(data["shapes"], list):
#         filtered_shapes = []
#         for shape in data["shapes"]:
#             # 检查是否有 group_id 字段且是否满足目标 group_id
#             if shape.get("group_id") == target_group:
#                 filtered_shapes.append(shape)
#         # 更新 shapes 数组
#         data["shapes"] = filtered_shapes

#     # 此处确保其它字段都保留（一般原始文件中会有如下字段）
#     # 提取信息示例（若需进一步处理可根据需要提取）
#     result = {
#         "version": data.get("version"),
#         "flags": data.get("flags"),
#         "shapes": data.get("shapes"),
#         "imagePath": data.get("imagePath"),
#         "imageData": data.get("imageData"),
#         "imageHeight": data.get("imageHeight"),
#         "imageWidth": data.get("imageWidth"),
#         "description": data.get("description")
#     }

#     # 将处理后的 JSON 数据写入到输出文件夹
#     try:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, ensure_ascii=False, indent=4)
#         print(f"处理并保存文件：{filename}（保留 group_id={target_group} 的项）")
#     except Exception as e:
#         print(f"写入 {filename} 时出错：{e}")

# print("全部处理完成！")


############合并文件夹
# import os
# import shutil

# def merge_and_rename(img_folder1, img_folder2, json_folder1, json_folder2, out_img_folder, out_json_folder):
#     # 如果输出文件夹不存在则创建
#     os.makedirs(out_img_folder, exist_ok=True)
#     os.makedirs(out_json_folder, exist_ok=True)
    
#     counter = 0  # 计数器，从0开始

#     # 定义一个处理单个文件夹的函数
#     def process_folder(img_folder, json_folder):
#         nonlocal counter
#         # 获取图片文件（假设扩展名为 .jpg）
#         img_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')])
        
#         for img_file in img_files:
#             # 定义新文件名
#             new_img_name = f"frame_{counter:05d}.jpg"
#             new_json_name = f"frame_{counter:05d}.json"
            
#             src_img_path = os.path.join(img_folder, img_file)
#             dst_img_path = os.path.join(out_img_folder, new_img_name)
            
#             # 复制图片文件
#             shutil.copy(src_img_path, dst_img_path)
#             print(f"复制图片: {src_img_path} ---> {dst_img_path}")
            
#             # 构造对应 JSON 文件名，假设 JSON 文件名与图片名称对应（jpg -> json）
#             json_file = os.path.splitext(img_file)[0] + ".json"
#             src_json_path = os.path.join(json_folder, json_file)
#             dst_json_path = os.path.join(out_json_folder, new_json_name)
            
#             # 检查 JSON 文件是否存在再复制
#             if os.path.exists(src_json_path):
#                 shutil.copy(src_json_path, dst_json_path)
#                 print(f"复制JSON : {src_json_path} ---> {dst_json_path}")
#             else:
#                 print(f"警告：找不到对应的JSON文件 {src_json_path}")
            
#             counter += 1

#     # 依次处理两个文件夹
#     process_folder(img_folder1, json_folder1)
#     process_folder(img_folder2, json_folder2)
    
#     print("合并完成，总共复制了 {} 张图片和对应 JSON 文件".format(counter))

# if __name__ == "__main__":
#     # 请在下面修改成你的实际文件夹路径
#     img_folder1 = r"Z:/Badminton_HAR_O/train/CLEAN/0227_lindan09_left_clip1_0-150_images1"      # 第一个图片文件夹路径
#     img_folder2 = r"Z:/Badminton_HAR_O/train/CLEAN/0227_lindan09_left_clip1_0-150_images2"      # 第二个图片文件夹路径
#     json_folder1 = r"Z:/Badminton_HAR_O/train/CLEAN/checkcheck1"       # 第一个json文件夹路径
#     json_folder2 = r"Z:/Badminton_HAR_O/train/CLEAN/checkcheck2"        # 第二个json文件夹路径
#     out_img_folder = r"C:/Users/Administrator/Desktop/data_union/images"  # 合并后图片的输出文件夹
#     out_json_folder = r"C:/Users/Administrator/Desktop/data_union/labels_json"  # 合并后json的输出文件夹
    
#     merge_and_rename(img_folder1, img_folder2, json_folder1, json_folder2, out_img_folder, out_json_folder)


# # # ########目标裁剪
# import os
# import json
# from PIL import Image

# def crop_and_update_json(image_path, json_path, out_image_path, out_json_path):
#     # 读取图片
#     image = Image.open(image_path)

#     # 读取 JSON 数据
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # 从 shapes 中查找目标框（person） ——假设只有一个目标框
#     person_box = None
#     for shape in data['shapes']:
#         if shape.get("label") == "person" and shape.get("shape_type") == "rectangle":
#             # 矩形框中的 points 顺序为：左上、右上、右下、左下
#             person_box = shape["points"]
#             break
    
#     if person_box is None:
#         raise ValueError("未在 JSON 文件中找到 'person' 的 rectangle 对象。")

#     # 计算裁剪区域的边界：左上角、右下角
#     # 注意：这里假设坐标为浮点数，可以对其取整
#     xs = [pt[0] for pt in person_box]
#     ys = [pt[1] for pt in person_box]
#     left, top = int(min(xs)), int(min(ys))
#     right, bottom = int(max(xs)), int(max(ys))

#     # 裁剪图片
#     cropped_image = image.crop((left, top, right, bottom))

#     # 对 json 文件里的所有点坐标进行修改：减去裁剪区域左上角的偏移
#     for shape in data["shapes"]:
#         # points 可能是单个点或者多个点
#         new_points = []
#         for pt in shape["points"]:
#             # 根据点的格式来处理，这里假设都是列表格式 [x, y]
#             new_x = pt[0] - left
#             new_y = pt[1] - top
#             new_points.append([new_x, new_y])
#         shape["points"] = new_points

#     # 更新 imagePath 和 imageWidth, imageHeight 字段
#     data["imagePath"] = os.path.basename(image_path)
#     data["imageWidth"] = cropped_image.width
#     data["imageHeight"] = cropped_image.height

#     # 保存裁剪后的图片和更新后的 json 文件
#     cropped_image.save(out_image_path)
#     with open(out_json_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
#     # 输入文件夹和输出文件夹路径（可以根据你的实际目录结构修改）
#     input_img_folder = r"C:/Users/Administrator/Desktop/data_union/images" 
#     input_json_folder = r"C:/Users/Administrator/Desktop/data_union/labels_json"
#     output_img_folder = r"C:/Users/Administrator/Desktop/data/images" 
#     output_json_folder = r"C:/Users/Administrator/Desktop/data/labels_json"
#     # 确保输出文件夹存在，如果不存在则创建
#     os.makedirs(output_img_folder, exist_ok=True)
#     os.makedirs(output_json_folder, exist_ok=True)

#     # 遍历所有 JSON 文件（假设图片和 JSON 文件名称对应）
#     for json_filename in os.listdir(input_json_folder):
#         if not json_filename.endswith('.json'):
#             continue

#         json_path = os.path.join(input_json_folder, json_filename)
#         # 假设图片文件名与json文件同名但后缀为 .jpg
#         image_filename = json_filename.replace('.json', '.jpg')
#         image_path = os.path.join(input_img_folder, image_filename)

#         # 定义输出路径
#         out_image_path = os.path.join(output_img_folder, image_filename)
#         out_json_path = os.path.join(output_json_folder, json_filename)

#         try:
#             crop_and_update_json(image_path, json_path, out_image_path, out_json_path)
#             print(f"成功处理文件：{image_filename} 和 {json_filename}")
#         except Exception as e:
#             print(f"处理 {json_filename} 时出错: {e}")




# # # # # # #######检查是否有17个关键点
# import os
# import json


# def check_keypoints_in_json_folder(json_folder):
#     for root, dirs, files in os.walk(json_folder):
#         for file in files:
#             if file.endswith('.json'):
#                 json_path = os.path.join(root, file)
#                 try:
#                     with open(json_path, 'r') as f:
#                         data = json.load(f)
#                     # 统计关键点数量
#                     keypoint_count = sum(1 for shape in data['shapes'] if shape['label'] != 'person')
#                     if keypoint_count == 17:
#                         print(f"{json_path} 包含 17 个关键点。")
#                     else:
#                         print(f"{json_path} 包含 {keypoint_count} 个关键点，不等于 17 个。")
#                 except Exception as e:
#                     print(f"处理 {json_path} 时出错: {e}")


# if __name__ == "__main__":
#     json_folder = r"C:/Users/Administrator/Desktop/900/qing"    # 替换为你的 JSON 文件夹路径
#     check_keypoints_in_json_folder(json_folder)

##json换txt文件
# import os
# import json

# def convert_json_to_yolov8_pose(json_file, output_file):
#     # 打开 json 文件
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 获取图像尺寸（用于归一化）  
#     # 注意：需保证 json 中提供的 imageWidth 与 imageHeight为正确的数值
#     img_w = data.get("imageWidth")
#     img_h = data.get("imageHeight")
#     if not img_w or not img_h:
#         print(f"文件 {json_file} 缺少 imageWidth/imageHeight 信息")
#         return

#     shapes = data.get("shapes", [])
    
#     # 提取 "person" 的目标框（rectangle），此处假定只有一个 person 框
#     bbox = None
#     for shape in shapes:
#         if shape.get("shape_type") == "rectangle" and shape.get("label") == "person":
#             # 假定 points 为矩形的四个顶点，顺序为：左上、右上、右下、左下
#             pts = shape.get("points", [])
#             if len(pts) >= 2:
#                 x_coords = [pt[0] for pt in pts]
#                 y_coords = [pt[1] for pt in pts]
#                 xmin = min(x_coords)
#                 xmax = max(x_coords)
#                 ymin = min(y_coords)
#                 ymax = max(y_coords)
#                 # 计算中心点、宽度、高度（归一化）
#                 center_x = ((xmin + xmax) / 2.0) / img_w
#                 center_y = ((ymin + ymax) / 2.0) / img_h
#                 width = (xmax - xmin) / img_w
#                 height = (ymax - ymin) / img_h
#                 bbox = (center_x, center_y, width, height)
#             break

#     if bbox is None:
#         print(f"文件 {json_file} 中未找到 person 的目标框")
#         return

#     # 定义关键点的标准顺序（这里以 COCO 为例）
#     keypoint_order = [
#         "nose",
#         "left_eye",
#         "right_eye",
#         "left_ear",
#         "right_ear",
#         "left_shoulder",
#         "right_shoulder",
#         "left_elbow",
#         "right_elbow",
#         "left_wrist",
#         "right_wrist",
#         "left_hip",
#         "right_hip",
#         "left_knee",
#         "right_knee",
#         "left_ankle",
#         "right_ankle"
#     ]
    
#     # 收集关键点信息
#     keypoints_dict = {}
#     for shape in shapes:
#         if shape.get("shape_type") == "point":
#             label = shape.get("label")
#             pts = shape.get("points", [])
#             if pts and len(pts) > 0:
#                 keypoints_dict[label] = pts[0]  # pts[0] 为 (x, y)

#     # 按照标准顺序，转换坐标并统一可见度为 1
#     keypoints_out = []
#     for kpt in keypoint_order:
#         if kpt in keypoints_dict:
#             x, y = keypoints_dict[kpt]
#         else:
#             # 如果缺失关键点，可以设为 0 (也可以根据需要调整)
#             x, y = 0.0, 0.0
#         keypoints_out.append(x / img_w)
#         keypoints_out.append(y / img_h)
#         keypoints_out.append(1.0)  # 可见度固定为1

#     # 按要求生成一行 txt 内容，数值保留6位小数
#     # 此处类别固定为0 (person)
#     elements = [format(0, '.0f')]
#     elements += [format(v, '.6f') for v in bbox]
#     elements += [format(v, '.6f') for v in keypoints_out]
#     line = " ".join(elements)

#     # 写入 txt 文件
#     with open(output_file, 'w', encoding='utf-8') as out_f:
#         out_f.write(line + "\n")


# def batch_convert(input_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 遍历 input_dir 中的 json 文件
#     for file_name in os.listdir(input_dir):
#         if file_name.lower().endswith('.json'):
#             json_path = os.path.join(input_dir, file_name)
#             # 输出文件名将 json 后缀替换成 txt
#             txt_name = os.path.splitext(file_name)[0] + '.txt'
#             output_path = os.path.join(output_dir, txt_name)
#             convert_json_to_yolov8_pose(json_path, output_path)
#             print(f"已转换：{json_path} -> {output_path}")


# if __name__ == "__main__":
#     # 请设置好你的 json 文件所在的目录和 txt 文件输出目录
#     input_dir = r"C:/Users/Administrator/Desktop/badmin900/json"   # 修改为你的 json 文件夹路径
#     output_dir = r"C:/Users/Administrator/Desktop/badmin900/labels"     # 修改为你希望输出的 txt 文件夹路径
#     batch_convert(input_dir, output_dir)

####划分数据集8：2
# import os
# import shutil
# import random

# # 数据集的原始文件夹路径
# images_dir = r"C:/Users/Administrator/Desktop/badmin900/images"
# labels_dir = r"C:/Users/Administrator/Desktop/badmin900/labels"

# # 目标文件夹（YOLO 格式）下的子文件夹路径
# train_images_dir = os.path.join(images_dir, "train")
# val_images_dir = os.path.join(images_dir, "val")
# train_labels_dir = os.path.join(labels_dir, "train")
# val_labels_dir = os.path.join(labels_dir, "val")

# # 如果目标文件夹不存在，则创建
# for folder in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#         print(f"创建文件夹：{folder}")

# # 获取所有图片文件列表（假设图片格式为常见的扩展名，可根据需要修改）
# valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
# image_files = [f for f in os.listdir(images_dir)
#                if os.path.isfile(os.path.join(images_dir, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

# # 打乱顺序
# random.shuffle(image_files)

# # 按 8:2 划分
# num_train = int(len(image_files) * 0.8)
# train_files = image_files[:num_train]
# val_files = image_files[num_train:]

# print(f"总共 {len(image_files)} 张图片，训练集：{len(train_files)} 张，验证集：{len(val_files)} 张")

# # 移动文件到对应的子文件夹
# def move_files(file_list, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
#     for file in file_list:
#         # 移动图片文件
#         src_img_path = os.path.join(src_images_dir, file)
#         dst_img_path = os.path.join(dst_images_dir, file)
#         shutil.move(src_img_path, dst_img_path)
        
#         # 对应的 label 文件（假设为 txt 格式）
#         file_name, _ = os.path.splitext(file)
#         label_file = file_name + ".txt"
#         src_label_path = os.path.join(src_labels_dir, label_file)
#         if os.path.exists(src_label_path):
#             dst_label_path = os.path.join(dst_labels_dir, label_file)
#             shutil.move(src_label_path, dst_label_path)
#         else:
#             print(f"警告：{src_label_path} 不存在！")

# # 将训练集文件移动
# move_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
# # 将验证集文件移动
# move_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)

# print("数据集划分完成！")

######将txt标签中的2.000000 替换为 1.000000
# import os
# import glob

# label_files = glob.glob("C:/Users/Administrator/Desktop/badminton_data/labels/val/*.txt")

# for file in label_files:
#     with open(file, 'r') as f:
#         content = f.read()
    
#     # 将所有的 2.000000 替换为 1.000000
#     modified_content = content.replace(' 2.000000', ' 1.000000')
    
#     with open(file, 'w') as f:
#         f.write(modified_content)
    
#     print(f"处理文件: {file}")

# print("所有文件处理完毕")

#####视频重命名
# import os

# def rename_videos(folder_path):
#     """
#     将指定文件夹中的视频文件按照 `001.mp4`, `002.mp4`, `003.mp4`, ... 的格式重新命名。
#     """
#     # 获取文件夹中的所有文件
#     files = os.listdir(folder_path)
#     videos = [f for f in files if f.endswith(".mp4")]  # 只处理 mp4 格式的视频，可根据需要修改

#     # 按照顺序重命名视频文件
#     for index, video in enumerate(videos):
#         # 构建新文件名，例如 001.mp4, 002.mp4, ...
#         new_name = f"{index + 1:03d}.mp4"
#         # 构建完整的文件路径
#         old_path = os.path.join(folder_path, video)
#         new_path = os.path.join(folder_path, new_name)
#         # 重命名文件
#         os.rename(old_path, new_path)
#         print(f"Renamed {video} to {new_name}")

# if __name__ == "__main__":
#     # 指定视频文件夹路径
#     folder_path = r"Z:/视频素材/2025.4.15 侧边678-2k30-下午"
#     rename_videos(folder_path)

# import os

# # 请将此路径修改为你的视频文件夹所在路径
# VIDEO_FOLDER = r"Z:/视频素材/2025.4.15 侧边678-2k30-上午"

# # 获取视频文件夹中的所有文件
# video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]

# # 提取数字部分并排序
# video_numbers = sorted([int(f.split('.')[0]) for f in video_files])

# # 重命名文件
# for index, number in enumerate(video_numbers, start=1):
#     old_name = f"{number}.mp4"
#     new_name = f"{index:03d}.mp4"  # 新的文件名，从001开始
#     old_path = os.path.join(VIDEO_FOLDER, old_name)
#     new_path = os.path.join(VIDEO_FOLDER, new_name)
    
#     try:
#         os.rename(old_path, new_path)
#         print(f"Renamed '{old_name}' ➔ '{new_name}'")
#     except FileNotFoundError:
#         print(f"Error: '{old_name}' not found in the folder.")
#     except Exception as e:
#         print(f"Failed to rename '{old_name}': {e}")

# print("重命名完成。")



# import os
# import json
# import shutil

# def export_renamed_annotations(src_folder, dst_folder):
#     """
#     将 src_folder 中的 JPG + JSON 文件批量重命名为 frame_00000.jpg/.json ~ frame_NNNNN.jpg/.json，
#     并将重命名后的图片和更新了 imagePath 的 JSON 保存到 dst_folder。

#     - src_folder: 原始图片和 JSON 文件所在目录
#     - dst_folder: 输出重命名文件的目标目录
#     """
#     os.makedirs(dst_folder, exist_ok=True)

#     # 按名称排序所有 JPG
#     jpg_files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith('.jpg')])

#     for new_idx, old_jpg in enumerate(jpg_files):
#         old_base = os.path.splitext(old_jpg)[0]
#         old_json = old_base + '.json'

#         # 构造新文件名
#         new_base = f"frame_{new_idx:05d}"
#         new_jpg = new_base + '.jpg'
#         new_json = new_base + '.json'

#         # 1) 复制并重命名图片到目标文件夹
#         src_jpg_path = os.path.join(src_folder, old_jpg)
#         dst_jpg_path = os.path.join(dst_folder, new_jpg)
#         shutil.copy2(src_jpg_path, dst_jpg_path)
#         print(f"Copied image: {old_jpg} -> {new_jpg}")

#         # 2) 如果存在对应 JSON，更新 imagePath 并写入目标文件夹
#         src_json_path = os.path.join(src_folder, old_json)
#         dst_json_path = os.path.join(dst_folder, new_json)

#         if os.path.exists(src_json_path):
#             with open(src_json_path, 'r', encoding='utf-8') as jf:
#                 data = json.load(jf)
#             data['imagePath'] = new_jpg
#             with open(dst_json_path, 'w', encoding='utf-8') as jf:
#                 json.dump(data, jf, ensure_ascii=False, indent=2)
#             print(f"Created JSON: {new_json} (imagePath='{new_jpg}')")
#         else:
#             print(f"Warning: 未找到对应 JSON 于 {old_jpg}，已跳过 JSON。")

# if __name__ == '__main__':
#     # —— 修改为你的源目录和目标目录 ——  
#     src_folder = r"Z:/Badminton_HAR_O/train/jian4.15 00011-00311"  # 原始文件夹
#     dst_folder = r"C:/Users/Administrator/Desktop/jian"        # 新目标文件夹
#     export_renamed_annotations(src_folder, dst_folder)





# import os
# import json
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image

# def display_images_with_annotations(src_folder, num_images=5):
#     # 1. 找到所有 .jpg 并排序
#     jpg_files = sorted(f for f in os.listdir(src_folder) if f.lower().endswith('.jpg'))
#     if not jpg_files:
#         print("目录中没有找到任何 .jpg 文件。")
#         return

#     # 2. 只处理前 num_images 张
#     to_show = jpg_files[:min(num_images, len(jpg_files))]

#     for idx, img_name in enumerate(to_show, 1):
#         # 对应的 JSON 文件
#         base = os.path.splitext(img_name)[0]
#         json_name = base + ".json"
#         img_path = os.path.join(src_folder, img_name)
#         json_path = os.path.join(src_folder, json_name)

#         if not os.path.exists(json_path):
#             print(f"跳过 {img_name}，找不到对应的 {json_name}")
#             continue

#         # 3. 加载图片和 JSON
#         img = Image.open(img_path)
#         with open(json_path, 'r') as f:
#             data = json.load(f)

#         # 4. 创建画布并显示图片
#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.imshow(img)
#         ax.set_title(f"{idx}. {img_name}")
#         ax.axis('off')

#         # 5. 遍历所有 shapes
#         for shape in data.get('shapes', []):
#             pts = shape.get('points', [])
#             label = shape.get('label', '')
#             stype = shape.get('shape_type', '')

#             if stype == 'rectangle' and len(pts) == 4:
#                 # rectangle: 用前两个点 (top‑left, top‑right) 计算 x,y,w,h
#                 x0, y0 = pts[0]
#                 x1, y1 = pts[1]
#                 w = abs(x1 - x0)
#                 h = abs(pts[2][1] - y0)
#                 rect = patches.Rectangle((x0, y0), w, h,
#                                          linewidth=2, edgecolor='r', facecolor='none')
#                 ax.add_patch(rect)
#                 ax.text(x0, y0 - 5, label, color='r', fontsize=10, weight='bold')

#             elif stype == 'point':
#                 # point: 逐个画绿点，并标注
#                 for (x, y) in pts:
#                     ax.plot(x, y, 'go', markersize=5)
#                     ax.text(x + 3, y - 3, label, color='g', fontsize=8)

#         plt.tight_layout()
#         plt.show()


# if __name__ == "__main__":
#     # —— 修改为你的文件夹路径 —— 
#     folder_path = r"C:/Users/Administrator/Desktop/jianjson"
#     display_images_with_annotations(folder_path, num_images=5)




# import os
# import shutil
# import json

# # 源文件夹列表，按合并顺序排列
# source_dirs =[r"C:/Users/Administrator/Desktop/900/jian", r"C:/Users/Administrator/Desktop/900/yellow",r"C:/Users/Administrator/Desktop/900/qing"]
# # 目标文件夹路径（你可以在这里直接修改路径）
# output_dir = r"C:/Users/Administrator/Desktop/900/all"
# # 支持的图片扩展名
# img_exts = ('.jpg', '.jpeg', '.png')

# # 创建目标文件夹（如果不存在的话）
# os.makedirs(output_dir, exist_ok=True)

# counter = 0
# for src in source_dirs:
#     # 列出并按名称排序，确保顺序一致
#     for fname in sorted(os.listdir(src)):
#         # 只处理图片文件
#         if not fname.lower().endswith(img_exts):
#             continue

#         # 构造源文件路径
#         img_src_path = os.path.join(src, fname)
#         # 对应的 JSON 文件名（假设与图片同名，只是后缀不同）
#         base_name, ext = os.path.splitext(fname)
#         json_src_path = os.path.join(src, base_name + '.json')

#         # 生成新的连续编号文件名，如 "00000.jpg"、"00001.jpg" …
#         new_base = f'{counter:05d}'
#         new_img_name = new_base + ext.lower()
#         new_json_name = new_base + '.json'

#         # 复制图片到目标文件夹并重命名
#         shutil.copy(img_src_path, os.path.join(output_dir, new_img_name))

#         # 读取 JSON，修改 imagePath 字段后写入目标文件夹
#         if os.path.exists(json_src_path):
#             with open(json_src_path, 'r', encoding='utf-8') as jf:
#                 data = json.load(jf)
#             # 更新 JSON 中记录的图片路径
#             data['imagePath'] = new_img_name
#             # 将修改后的 JSON 写入目标并重命名
#             with open(os.path.join(output_dir, new_json_name), 'w', encoding='utf-8') as jf:
#                 json.dump(data, jf, ensure_ascii=False, indent=4)
#         else:
#             print(f'Warning: 找不到对应的 JSON 文件 {json_src_path}')

#         counter += 1

# print(f'合并完成，生成了 {counter} 对图片+JSON 文件，保存在文件夹：{output_dir}')

######txt文件删除目标框信息
import os

def process_txt_file(input_path, output_path):
    """
    处理单个txt文件，删除目标框信息，只保留 class_id 和关键点坐标。
    
    :param input_path: 输入txt文件的路径
    :param output_path: 输出txt文件的路径
    """
    with open(input_path, 'r') as infile:
        lines = infile.readlines()
    
    processed_lines = []
    for line in lines:
        parts = line.strip().split()
        
        # 检查是否至少有关键点信息（前5个是目标框信息）
        if len(parts) > 5:
            class_id = parts[0]  # 保留类别 ID
            keypoints = parts[5:]  # 关键点信息从第6个元素开始
            processed_lines.append(f"{class_id} {' '.join(keypoints)}\n")
        else:
            # 如果没有关键点信息，跳过该行并记录警告
            print(f"Warning: Skipping invalid line in {input_path}: {line.strip()}")
            processed_lines.append('\n')  # 或者直接跳过（根据需求调整）
    
    # 将处理后的内容写入输出文件
    with open(output_path, 'w') as outfile:
        outfile.writelines(processed_lines)

def process_folder(input_folder, output_folder):
    """
    处理文件夹中的所有txt文件，删除目标框信息，只保留 class_id 和关键点坐标。
    
    :param input_folder: 包含 train 和 val 文件夹的输入目录
    :param output_folder: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历 train 和 val 文件夹
    for subset in ['train', 'val']:
        input_subset_folder = os.path.join(input_folder, subset)
        output_subset_folder = os.path.join(output_folder, subset)
        
        # 创建对应的输出子文件夹
        os.makedirs(output_subset_folder, exist_ok=True)
        
        # 遍历该子文件夹中的所有txt文件
        for filename in os.listdir(input_subset_folder):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(input_subset_folder, filename)
                output_file_path = os.path.join(output_subset_folder, filename)
                
                # 处理单个txt文件
                print(f"Processing file: {input_file_path}")
                process_txt_file(input_file_path, output_file_path)

# 使用示例
input_folder = r"C:/Users/Administrator/Desktop/badminton_data/labels"  # 替换为你的输入文件夹路径
output_folder = r"C:/Users/Administrator/Desktop/badminton/labels"  # 替换为你希望保存的输出文件夹路径

process_folder(input_folder, output_folder)