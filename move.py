###################### 分类时刻和林丹数据 ######################
# import os
# import shutil
# from pathlib import Path

# # 源文件夹和目标文件夹路径
# source_root = r"Z:/BadData/Benchmark/Benchmark_pose/7995/rawdata"
# shike_dest = r"Z:/BadData/Benchmark/Benchmark_pose/7995/shike_raw"
# lindan_dest = r"Z:/BadData/Benchmark/Benchmark_pose/7995/lindan_raw"

# # 要复制的文件夹类型
# folder_types = ["images", "annotations", "labels"]

# # 要复制到shike_raw的文件夹
# shike_left_folders = ["1", "2", "3", "4", "5", "7", "8", "9", "10", "11", "12", "13", "14"]
# shike_right_folders = ["1", "2", "3", "4", "5", "6", "7", "10", "11", "12", "13", "14", "15", "16"]
# shike_back_folders = ["2", "3"]

# def copy_folders(src_base, dest_base, view, folders_to_copy, copy_to_shike=True):
#     """复制指定视角下的文件夹到目标位置"""
#     for folder_type in folder_types:
#         # 修正：所有文件夹类型都包含dataset子目录
#         src_path = os.path.join(src_base, folder_type, "dataset", view)
        
#         # 确定目标路径
#         if copy_to_shike:
#             # 目标路径也应包含dataset子目录
#             dest_path = os.path.join(shike_dest, folder_type, "dataset", view)
#         else:
#             dest_path = os.path.join(lindan_dest, folder_type, "dataset", view)
            
#         # 确保目标路径存在
#         os.makedirs(dest_path, exist_ok=True)
        
#         # 获取源文件夹中的所有子文件夹
#         try:
#             all_folders = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f))]
#         except FileNotFoundError:
#             print(f"警告: 源路径不存在 {src_path}")
#             continue
            
#         # 确定要复制的文件夹
#         if copy_to_shike:
#             folders = folders_to_copy
#         else:
#             folders = [f for f in all_folders if f not in folders_to_copy]
            
#         # 复制文件夹
#         for folder in folders:
#             src_folder = os.path.join(src_path, folder)
#             dest_folder = os.path.join(dest_path, folder)
            
#             if os.path.exists(src_folder):
#                 print(f"复制 {src_folder} 到 {dest_folder}")
#                 if os.path.exists(dest_folder):
#                     shutil.rmtree(dest_folder)
#                 shutil.copytree(src_folder, dest_folder)
#             else:
#                 print(f"警告: 源文件夹不存在 {src_folder}")

# def main():
#     # 确保目标文件夹结构存在
#     for dest in [shike_dest, lindan_dest]:
#         for folder_type in folder_types:
#             # 修正：所有文件夹都包含dataset子目录
#             os.makedirs(os.path.join(dest, folder_type, "dataset", "left"), exist_ok=True)
#             os.makedirs(os.path.join(dest, folder_type, "dataset", "right"), exist_ok=True)
#             os.makedirs(os.path.join(dest, folder_type, "dataset", "back"), exist_ok=True)
    
#     # 复制到shike_raw
#     copy_folders(source_root, shike_dest, "left", shike_left_folders, True)
#     copy_folders(source_root, shike_dest, "right", shike_right_folders, True)
#     copy_folders(source_root, shike_dest, "back", shike_back_folders, True)
    
#     # 复制到lindan_raw
#     copy_folders(source_root, lindan_dest, "left", shike_left_folders, False)
#     copy_folders(source_root, lindan_dest, "right", shike_right_folders, False)
#     copy_folders(source_root, lindan_dest, "back", shike_back_folders, False)
    
#     print("文件复制完成!")

# if __name__ == "__main__":
#     main()





################### 统计林丹时刻每个视角的数据数量和总和 ##########################

# import os
# from pathlib import Path

# def count_images(directory):
#     """计算指定目录下所有图片文件的数量"""
#     image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
#     count = 0
    
#     if not os.path.exists(directory):
#         print(f"警告: 路径不存在 {directory}")
#         return 0
    
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if any(file.lower().endswith(ext) for ext in image_extensions):
#                 count += 1
    
#     return count

# def main():
#     # 基础路径
#     lindan_base = r"Z:/BadData/Benchmark/Benchmark_pose/7995/lindan_raw/images"
#     shike_base = r"Z:/BadData/Benchmark/Benchmark_pose/7995/shike_raw/images"
    
#     # 检查路径是否存在
#     if not os.path.exists(lindan_base):
#         print(f"错误: 路径不存在 {lindan_base}")
#         return
    
#     if not os.path.exists(shike_base):
#         print(f"错误: 路径不存在 {shike_base}")
#         return
    
#     # 统计lindan_raw的图片数量
#     lindan_left_path = os.path.join(lindan_base, "left")
#     lindan_right_path = os.path.join(lindan_base,  "right")
#     lindan_back_path = os.path.join(lindan_base,  "back")
    
#     lindan_left_count = count_images(lindan_left_path)
#     lindan_right_count = count_images(lindan_right_path)
#     lindan_back_count = count_images(lindan_back_path)
#     lindan_total = lindan_left_count + lindan_right_count + lindan_back_count
    
#     # 统计lindan_raw每个视角下子文件夹的图片数量
#     lindan_left_subfolders = {}
#     lindan_right_subfolders = {}
#     lindan_back_subfolders = {}
    
#     if os.path.exists(lindan_left_path):
#         for subfolder in os.listdir(lindan_left_path):
#             subfolder_path = os.path.join(lindan_left_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 lindan_left_subfolders[subfolder] = count_images(subfolder_path)
    
#     if os.path.exists(lindan_right_path):
#         for subfolder in os.listdir(lindan_right_path):
#             subfolder_path = os.path.join(lindan_right_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 lindan_right_subfolders[subfolder] = count_images(subfolder_path)
    
#     if os.path.exists(lindan_back_path):
#         for subfolder in os.listdir(lindan_back_path):
#             subfolder_path = os.path.join(lindan_back_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 lindan_back_subfolders[subfolder] = count_images(subfolder_path)
    
#     # 统计shike_raw的图片数量
#     shike_left_path = os.path.join(shike_base, "left")
#     shike_right_path = os.path.join(shike_base,  "right")
#     shike_back_path = os.path.join(shike_base,  "back")
    
#     shike_left_count = count_images(shike_left_path)
#     shike_right_count = count_images(shike_right_path)
#     shike_back_count = count_images(shike_back_path)
#     shike_total = shike_left_count + shike_right_count + shike_back_count
    
#     # 统计shike_raw每个视角下子文件夹的图片数量
#     shike_left_subfolders = {}
#     shike_right_subfolders = {}
#     shike_back_subfolders = {}
    
#     if os.path.exists(shike_left_path):
#         for subfolder in os.listdir(shike_left_path):
#             subfolder_path = os.path.join(shike_left_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 shike_left_subfolders[subfolder] = count_images(subfolder_path)
    
#     if os.path.exists(shike_right_path):
#         for subfolder in os.listdir(shike_right_path):
#             subfolder_path = os.path.join(shike_right_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 shike_right_subfolders[subfolder] = count_images(subfolder_path)
    
#     if os.path.exists(shike_back_path):
#         for subfolder in os.listdir(shike_back_path):
#             subfolder_path = os.path.join(shike_back_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 shike_back_subfolders[subfolder] = count_images(subfolder_path)
    
#     # 打印结果
#     print("林丹数据集统计:")
#     print(f"  左视角: {lindan_left_count} 张图片")
#     for subfolder, count in sorted(lindan_left_subfolders.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
#         print(f"    - 子文件夹 {subfolder}: {count} 张图片")
    
#     print(f"  右视角: {lindan_right_count} 张图片")
#     for subfolder, count in sorted(lindan_right_subfolders.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
#         print(f"    - 子文件夹 {subfolder}: {count} 张图片")
    
#     print(f"  后视角: {lindan_back_count} 张图片")
#     for subfolder, count in sorted(lindan_back_subfolders.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
#         print(f"    - 子文件夹 {subfolder}: {count} 张图片")
    
#     print(f"  总计: {lindan_total} 张图片")
#     print()
    
#     print("石克数据集统计:")
#     print(f"  左视角: {shike_left_count} 张图片")
#     for subfolder, count in sorted(shike_left_subfolders.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
#         print(f"    - 子文件夹 {subfolder}: {count} 张图片")
    
#     print(f"  右视角: {shike_right_count} 张图片")
#     for subfolder, count in sorted(shike_right_subfolders.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
#         print(f"    - 子文件夹 {subfolder}: {count} 张图片")
    
#     print(f"  后视角: {shike_back_count} 张图片")
#     for subfolder, count in sorted(shike_back_subfolders.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
#         print(f"    - 子文件夹 {subfolder}: {count} 张图片")
    
#     print(f"  总计: {shike_total} 张图片")
#     print()
    
#     print(f"两个数据集总计: {lindan_total + shike_total} 张图片")

# if __name__ == "__main__":
#     main()







############## 将shike/lindan的数据合并为一个文件夹 ##############
# import os
# import shutil

# # 原始路径
# raw_base = r"Z:/BadData/Benchmark/Benchmark_pose/7995/shike_raw"
# raw_images = os.path.join(raw_base, "images")  # 添加dataset子目录
# raw_labels = os.path.join(raw_base, "labels")  # 添加dataset子目录

# # 目标路径
# target_base = r"Z:/BadData/Benchmark/Benchmark_pose/7995/shike" 
# target_images = os.path.join(target_base, "images")
# target_labels = os.path.join(target_base, "labels")

# # 创建目标文件夹
# os.makedirs(target_images, exist_ok=True)
# os.makedirs(target_labels, exist_ok=True)

# # 要跳过的子文件夹相对路径 - 使用完整路径
# skip_folders = [
#     os.path.normpath(os.path.join("left", "2")),
#     os.path.normpath(os.path.join("right", "1"))
# ]

# # 收集所有待处理的图片文件
# image_paths = []
# for side in ["left", "right", "back"]:
#     side_dir = os.path.join(raw_images, side)
#     if not os.path.exists(side_dir):
#         print(f"警告: 目录不存在 {side_dir}")
#         continue
        
#     for subfolder in os.listdir(side_dir):
#         subfolder_path = os.path.join(side_dir, subfolder)
#         if not os.path.isdir(subfolder_path):
#             continue
            
#         # 检查是否是要跳过的文件夹
#         rel_path = os.path.normpath(os.path.join(side, subfolder))
#         if rel_path in skip_folders:
#             print(f"跳过目录: {subfolder_path}")
#             continue
            
#         # 处理子文件夹中的图片
#         for root, _, files in os.walk(subfolder_path):
#             for fname in files:
#                 if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#                     image_paths.append(os.path.join(root, fname))

# # 按文件名排序
# image_paths.sort()

# # 开始复制并重命名
# for idx, img_path in enumerate(image_paths):
#     # 构造新文件名
#     ext = os.path.splitext(img_path)[1]
#     new_name = f"{idx}{ext}"
    
#     # 复制图片
#     dst_img = os.path.join(target_images, new_name)
#     shutil.copy2(img_path, dst_img)
    
#     # 对应的 label 文件路径
#     rel_path = os.path.relpath(img_path, raw_images)
#     lbl_path = os.path.join(raw_labels, os.path.splitext(rel_path)[0] + ".txt")
    
#     if os.path.exists(lbl_path):
#         dst_lbl = os.path.join(target_labels, f"{idx}.txt")
#         shutil.copy2(lbl_path, dst_lbl)
#     else:
#         print(f"警告: 找不到标注文件: {lbl_path}")
        
# print(f"全部复制并重命名完成。共处理了 {len(image_paths)} 张图片。")










################# 将lindan和shike数据合并 #################
# import os
# import shutil

# # 源数据集根路径
# base = r"Z:/BadData/Benchmark/Benchmark_pose/7995/hebing"  # 修改为正确的基础路径
# datasets = ["lindan", "shike"]

# # 目标路径
# target = os.path.join(os.path.dirname(base), "lin_shi")  # 在hebing的同级创建lin_shi
# target_images = os.path.join(target, "images")
# target_labels = os.path.join(target, "labels")

# # 创建目标文件夹
# os.makedirs(target_images, exist_ok=True)
# os.makedirs(target_labels, exist_ok=True)

# # 汇总所有图片文件的完整路径
# all_image_paths = []
# for ds in datasets:
#     img_dir = os.path.join(base, ds, "images")
#     if not os.path.exists(img_dir):
#         print(f"警告: 目录不存在 {img_dir}")
#         continue
        
#     # 直接列出目录中的所有文件
#     for fname in os.listdir(img_dir):
#         full_path = os.path.join(img_dir, fname)
#         if os.path.isfile(full_path) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#             all_image_paths.append(full_path)

# # 可选：按名称排序，保证每次结果一致
# all_image_paths.sort()

# # 依次复制并重命名
# count_success = 0
# for idx, img_path in enumerate(all_image_paths):
#     # 检查文件是否存在
#     if not os.path.exists(img_path):
#         print(f"警告: 图片文件不存在 {img_path}")
#         continue
        
#     ext = os.path.splitext(img_path)[1]
#     new_img_name = f"{idx}{ext}"
#     dst_img = os.path.join(target_images, new_img_name)
    
#     try:
#         shutil.copy2(img_path, dst_img)
#         # 构造对应的 label 路径并复制
#         # 从图片路径直接推导标签路径
#         img_filename = os.path.basename(img_path)
#         img_name_no_ext = os.path.splitext(img_filename)[0]
        
#         # 确定数据集名称
#         if "lindan" in img_path:
#             dataset_name = "lindan"
#         else:
#             dataset_name = "shike"
            
#         lbl_path = os.path.join(base, dataset_name, "labels", f"{img_name_no_ext}.txt")
        
#         if os.path.exists(lbl_path):
#             new_lbl_name = f"{idx}.txt"
#             dst_lbl = os.path.join(target_labels, new_lbl_name)
#             shutil.copy2(lbl_path, dst_lbl)
#             count_success += 1
#         else:
#             print(f"警告: 找不到标签文件 {lbl_path}")
            
#     except Exception as e:
#         print(f"处理失败 {img_path}: {e}")

# print(f"合并完成。成功处理 {count_success} 张图片及其标签，总共尝试处理 {len(all_image_paths)} 张图片。")








# # ############# 划分数据集 #############
# import os
# import shutil
# import random

# # 设置随机种子以保证可复现
# random.seed(42)

# # 原始路径
# base_dir = r"Z:/BadData/Benchmark/Benchmark_pose/7995/lin_shi"
# images_dir = os.path.join(base_dir, "images")
# labels_dir = os.path.join(base_dir, "labels")

# # 目标子文件夹
# splits = ["train", "val"]
# for split in splits:
#     os.makedirs(os.path.join(images_dir, split), exist_ok=True)
#     os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# # 收集所有图片文件名（只取文件名，不含路径）
# all_images = [f for f in os.listdir(images_dir)
#               if os.path.isfile(os.path.join(images_dir, f))
#               and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}]

# # 随机打乱并划分
# random.shuffle(all_images)
# n_total = len(all_images)
# n_train = int(n_total * 0.9)

# train_files = all_images[:n_train]
# val_files   = all_images[n_train:]

# # 定义一个小函数来复制对应的图片和 label
# def copy_split(file_list, split_name):
#     for img_name in file_list:
#         base, ext = os.path.splitext(img_name)
#         src_img = os.path.join(images_dir, img_name)
#         dst_img = os.path.join(images_dir, split_name, img_name)
#         shutil.copy2(src_img, dst_img)
        
#         src_lbl = os.path.join(labels_dir, base + ".txt")
#         dst_lbl = os.path.join(labels_dir, split_name, base + ".txt")
#         if os.path.exists(src_lbl):
#             shutil.copy2(src_lbl, dst_lbl)
#         else:
#             print(f"Warning: 找不到对应的 label 文件 -> {src_lbl}")

# # 复制到 train 和 val
# copy_split(train_files, "train")
# copy_split(val_files,   "val")

# print(f"总共图片：{n_total}，训练集：{len(train_files)}，验证集：{len(val_files)}")
# print("划分完成！")










# #####重定义labels#####
# import os

# # 你的 labels 所在文件夹
# labels_dir = r"Z:\BadData\Benchmark\Benchmark_pose\test\shike_test\labels"

# # 遍历所有 .txt 文件
# for fname in os.listdir(labels_dir):
#     if not fname.lower().endswith(".txt"):
#         continue
#     path = os.path.join(labels_dir, fname)
#     new_lines = []
    
#     with open(path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 6:
#                 # 少于 6 个字段就跳过（保护性判断）
#                 new_lines.append(line)
#                 continue

#             # 原 class_id
#             orig_cls = parts[0]
#             # 将 class_id 统一置为 0
#             parts[0] = "0"

#             # 如果原 class_id 不是 "1" 且不是 "2"，修正可见度
#             if orig_cls not in ("1", "2"):
#                 # 从第 5 个字段开始，后面都是 keypoints：x, y, v
#                 # 索引从 5 (0-based) 开始，每隔 3 个是一个 v，需要置为 "2"
#                 for i in range(5, len(parts), 3):
#                     parts[i+2] = "2"  # 第 i+2 项即 v

#             # 重组
#             new_lines.append(" ".join(parts) + "\n")

#     # 写回文件
#     with open(path, 'w') as f:
#         f.writelines(new_lines)

# print("所有 labels 文件处理完成。")


# import os

# # 修改为你 labels 文件夹的实际路径
# labels_dir = r"Z:\BadData\Benchmark\Benchmark_pose\test\lindan_test\labels"

# for fname in os.listdir(labels_dir):
#     if not fname.lower().endswith(".txt"):
#         continue

#     path_in = os.path.join(labels_dir, fname)
#     lines_out = []

#     with open(path_in, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             # YOLO-kpt 最少要有 6 个字段：cls,x,y,w,h, x1,y1,v1
#             if len(parts) < 6:
#                 lines_out.append(line)
#                 continue

#             # 从第 5 字段开始，每 3 个为一组 (x, y, v)
#             for i in range(5, len(parts), 3):
#                 # parts[i+2] 是可见度 v
#                 if parts[i+2] == '2':
#                     parts[i+2] = '0'

#             lines_out.append(" ".join(parts) + "\n")

#     # 覆盖写回
#     with open(path_in, 'w') as f:
#         f.writelines(lines_out)

# print("所有文件中，v=2 的关键点已改为 v=0。")







################### 修改关键点可见度 ####################
# import os
# from pathlib import Path

# # 要处理的目录
# txt_dir = Path('Z:/BadData/Benchmark/Benchmark_pose/7995/lin_shi/labels/val')

# # 遍历目录下所有 .txt 文件
# for txt_path in txt_dir.glob('*.txt'):
#     modified_lines = []
#     with txt_path.open('r', encoding='utf-8') as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 # 保持空行或格式异常行不变
#                 modified_lines.append(line)
#                 continue

#             class_id = parts[0]
#             # 如果 classid 不是 1 且不是 2，就修改可见度
#             if class_id not in ('1', '2'):
#                 # parts[1:5] 是边框坐标，不改动
#                 # parts[5:] 是 17*3 个关键点字段
#                 for idx in range(5 + 2, len(parts), 3):
#                     # 第三个字段（v）统一设为 '0'
#                     parts[idx] = '0'
#                 modified_lines.append(' '.join(parts) + '\n')
#             else:
#                 # classid 为 1 或 2，则保持原样
#                 modified_lines.append(line)

#     # 将修改后的内容写回文件（覆盖原文件）
#     with txt_path.open('w', encoding='utf-8') as f:
#         f.writelines(modified_lines)

# print("所有文件已处理完成。")









############### classid替换为 0 ########################
# import os
# from pathlib import Path

# # 要处理的目录
# txt_dir = Path('Z:/BadData/Benchmark/Benchmark_pose/7995/lin_shi/labels/val')

# # 遍历目录下所有 .txt 文件
# for txt_path in txt_dir.glob('*.txt'):
#     new_lines = []
#     with txt_path.open('r', encoding='utf-8') as f:
#         for line in f:
#             parts = line.rstrip('\n').split()
#             if not parts:
#                 # 空行保持不变
#                 new_lines.append(line)
#                 continue

#             # 将 classid（parts[0]）设为 '0'
#             parts[0] = '0'
#             # 重新拼接行，并加回换行符
#             new_lines.append(' '.join(parts) + '\n')

#     # 写回覆盖原文件
#     with txt_path.open('w', encoding='utf-8') as f:
#         f.writelines(new_lines)

# print("所有文件的 classid 已替换为 0。")











# #################可视化标注文件########################
# import os

# def find_txt_files_with_zero_keypoint_full_path(start_directory):
#     """
#     Traverses directories, reads YOLO format txt files, and finds the FULL PATH
#     of files where a line with class ID 1 or 2 contains a keypoint coordinate of 0.0.

#     Args:
#         start_directory (str): The path to the directory to start searching from.

#     Returns:
#         list: A list of full file paths that meet the criteria.
#     """
#     # Change variable name to reflect that we store full paths
#     matching_file_paths = []

#     # Ensure the start directory exists
#     if not os.path.isdir(start_directory):
#         print(f"Error: Directory not found: {start_directory}")
#         return matching_file_paths

#     print(f"Starting search in: {start_directory}")

#     # Walk through all directories and files in the start_directory
#     for root, _, files in os.walk(start_directory):
#         for filename in files:
#             # Process only text files
#             if filename.lower().endswith('.txt'):
#                 filepath = os.path.join(root, filename)
#                 # print(f"Checking file: {filepath}") # Optional: uncomment for verbose output

#                 found_condition_in_file = False
#                 try:
#                     with open(filepath, 'r', encoding='utf-8') as f:
#                         for line in f:
#                             line = line.strip()
#                             if not line: # Skip empty lines
#                                 continue

#                             parts = line.split()

#                             # A valid line for object + keypoints must have at least
#                             # class_id, cx, cy, w, h (5 parts) plus at least one keypoint (x, y, vis = 3 parts).
#                             # So, minimum total parts is 8 (class_id + 4 + 3). Let's be slightly more general
#                             # and check for at least 6 parts (class, cx, cy, w, h, kp1_x) to handle minimal cases,
#                             # although valid keypoints need sets of 3. A safer minimal check for *any* keypoint data
#                             # is checking if len(parts) > 5.
#                             if len(parts) <= 5:
#                                 continue # Not enough parts for class_id, box, and keypoints

#                             try:
#                                 class_id = int(parts[0])

#                                 # Check if the class ID is 1 or 2
#                                 if class_id in [1, 2]:
#                                     # Keypoint data starts from the 6th element (index 5).
#                                     # Keypoints are in sets of 3: x, y, visibility.
#                                     # We need to check indices 5, 6, 8, 9, 11, 12, ...
#                                     # These are indices i and i+1 for i starting at 5 and incrementing by 3.
#                                     keypoint_data_start_index = 5

#                                     # Ensure there are enough parts for at least one keypoint pair (x,y,vis)
#                                     if len(parts) > keypoint_data_start_index + 1: # Need at least index 5 (x) and 6 (y)

#                                         # Iterate through the keypoint coordinates (x and y)
#                                         # The keypoint data is structured as (x, y, visibility) repeated.
#                                         # We need to check the x and y values.
#                                         # Indices to check are 5, 6, 8, 9, 11, 12, ...
#                                         for i in range(keypoint_data_start_index, len(parts), 3):
#                                             # Ensure we don't go out of bounds when checking parts[i+1]
#                                             if i + 1 < len(parts):
#                                                 try:
#                                                     kp_x = float(parts[i])
#                                                     kp_y = float(parts[i+1])

#                                                     # Check if either coordinate is 0.0
#                                                     if kp_x == 0.0 or kp_y == 0.0:
#                                                         # Found a line matching the criteria in this file
#                                                         found_condition_in_file = True
#                                                         break # Found condition, no need to check more keypoints in this line

#                                                 except ValueError:
#                                                     # Handle cases where coordinate is not a valid number
#                                                     # print(f"Warning: Could not convert coordinate to float in file {filepath}, line: {line}")
#                                                     pass # Ignore lines with invalid numbers in coordinate position

#                                     if found_condition_in_file:
#                                         break # Found condition in this line, no need to check more lines in this file

#                             except ValueError:
#                                 # Handle cases where class_id is not a valid integer
#                                 # print(f"Warning: Could not convert class ID to int in file {filepath}, line: {line}")
#                                 pass # Ignore lines with invalid class ID

#                     # If the condition was met for any line in the file, add the FULL FILEPATH
#                     if found_condition_in_file:
#                          matching_file_paths.append(filepath)

#                 except Exception as e:
#                     print(f"Error reading file {filepath}: {e}")
#                     # Continue to the next file even if one fails

#     # Return the list of full file paths
#     return matching_file_paths

# # --- How to use the function ---

# # IMPORTANT: Replace this path with the actual path on your system
# start_path = r"Z:/BadData/Benchmark/Benchmark_pose/rawdata/longmao720/longmao720/labels/人体关键点以及拉框/right-20250512T064348Z-1-001/right"

# # Call the function
# # Change variable name to reflect that results are full paths
# result_file_paths = find_txt_files_with_zero_keypoint_full_path(start_path)

# # Print the results
# if result_file_paths:
#     print("\nFiles containing class ID 1 or 2 with a zero keypoint coordinate (Full Paths):")
#     # Iterate and print full paths
#     for file_path in result_file_paths:
#         print(file_path)
# else:
#     print("\nNo files found matching the criteria.")




import os
import cv2
from datetime import timedelta

def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 检查是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return 0
        
        # 获取视频帧数和帧率
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算时长（秒）
        duration = frame_count / fps if fps > 0 else 0
        
        # 释放视频对象
        cap.release()
        
        return duration
    
    except Exception as e:
        print(f"处理视频时出错 {video_path}: {e}")
        return 0

def format_duration(seconds):
    """将秒数格式化为 时:分:秒 格式"""
    return str(timedelta(seconds=int(seconds)))

def scan_videos_folder(base_dir):
    """扫描文件夹中的所有视频并获取时长"""
    # 支持的视频扩展名
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    # 结果存储
    video_durations = []
    
    # 遍历left和right文件夹
    for category in ['left', 'right']:
        category_path = os.path.join(base_dir, category)
        
        if not os.path.exists(category_path):
            print(f"警告: 目录不存在 {category_path}")
            continue
        
        # 遍历该目录下的所有文件
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
            # 检查是否是视频文件
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
                # 获取视频时长
                duration = get_video_duration(file_path)
                
                # 添加到结果列表
                video_durations.append({
                    'category': category,
                    'filename': filename,
                    'path': file_path,
                    'duration_seconds': duration,
                    'duration_formatted': format_duration(duration)
                })
    
    return video_durations

def main():
    # 视频文件夹路径
    videos_dir = r"Z:/BadData/Video/Amateur/double/doublevideo"  # 修改为你的视频文件夹路径
    
    print("正在扫描视频文件夹...")
    
    # 扫描视频
    video_durations = scan_videos_folder(videos_dir)
    
    # 按类别和文件名排序
    video_durations.sort(key=lambda x: (x['category'], x['filename']))
    
    # 打印结果
    print("\n===== 视频时长详情 =====\n")
    
    current_category = None
    for video in video_durations:
        # 如果类别变化，打印新类别标题
        if video['category'] != current_category:
            current_category = video['category']
            print(f"\n{current_category.upper()} 视频:")
        
        print(f"  {video['filename']}: {video['duration_formatted']}")
    
    # 统计总数
    left_videos = [v for v in video_durations if v['category'] == 'left']
    right_videos = [v for v in video_durations if v['category'] == 'right']
    
    print("\n===== 统计摘要 =====")
    print(f"Left视频数量: {len(left_videos)}")
    print(f"Right视频数量: {len(right_videos)}")
    print(f"总视频数量: {len(video_durations)}")
    
    # 计算总时长
    total_duration_left = sum(v['duration_seconds'] for v in left_videos)
    total_duration_right = sum(v['duration_seconds'] for v in right_videos)
    total_duration = total_duration_left + total_duration_right
    
    print(f"Left视频总时长: {format_duration(total_duration_left)}")
    print(f"Right视频总时长: {format_duration(total_duration_right)}")
    print(f"所有视频总时长: {format_duration(total_duration)}")

if __name__ == "__main__":
    main()
