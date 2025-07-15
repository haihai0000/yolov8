import json
import os
from PIL import Image

def convert_coco_to_anylabeling(coco_data):
    """
    将 COCO 格式的关键点检测 JSON 数据转换为 AnyLabeling/LabelMe 格式。

    Args:
        coco_data (dict): 遵循 COCO 格式的关键点检测 JSON 数据。

    Returns:
        dict: 转换后的 AnyLabeling/LabelMe 格式的 JSON 数据。
    """
    anylabeling_data = {
        "version": "2.5.4", # 可以根据实际AnyLabeling版本调整
        "flags": {},
        "shapes": []
    }

    # COCO 关键点顺序（17个关键点）
    keypoint_labels = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    for detection in coco_data.get("detections", []):
        bbox = detection["bbox"]
        # COCO bbox: [x_min, y_min, x_max, y_max] (您的示例是这种格式)
        # AnyLabeling bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (左上、右上、右下、左下)
        x_min, y_min, x_max, y_max = bbox
        
        # 将边界框添加到 shapes
        person_shape = {
            "kie_linking": [],
            "label": f"person_in{detection['track_id']}",
            "points": [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ],
            "group_id": detection["track_id"],
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {}
        }
        anylabeling_data["shapes"].append(person_shape)

        keypoints = detection["keypoints"]
        for i, kp in enumerate(keypoints):
            x, y, confidence = kp["x"], kp["y"], kp["confidence"]

            # 根据 x, y 和 confidence 确定 visibility
            # 如果 x, y 都是 0.0，通常表示未检测到或不可见，设置为 "0"
            # 如果 confidence > 0.5，设置为 "2" (可见)
            # 否则设置为 "1" (遮挡)
            if x == 0.0 and y == 0.0:
                visibility = "0"
            elif confidence > 0.5:
                visibility = "2"
            else:
                visibility = "1"
            
            # 将关键点添加到 shapes
            keypoint_shape = {
                "kie_linking": [],
                "label": keypoint_labels[i],
                "points": [[x, y]],
                "group_id": detection["track_id"],
                "description": None,
                "difficult": False,
                "shape_type": "point",
                "flags": {},
                "attributes": {
                    "visibility": visibility
                }
            }
            anylabeling_data["shapes"].append(keypoint_shape)

    return anylabeling_data

if __name__ == "__main__":
    # 在这里硬编码你的输入和输出路径
    input_path = r"Z:/double/left/1/jsons/frame_0.json" # JSON 输入路径（可以是文件或文件夹）
    output_folder = r"Z:/double/left/1" # 转换后的 JSON 输出根文件夹
    image_folder_path = r"Z:/double/left/1" # 对应的图片文件夹路径

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    def process_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                coco_input_data = json.load(f)
            
            converted_data = convert_coco_to_anylabeling(coco_input_data)
            
            # 获取 JSON 文件名（例如 frame_0.json）
            base_json_filename = os.path.basename(file_path)
            # 从 JSON 文件名（例如 'frame_0.json'）中，直接替换扩展名为 .jpg 来构建图片文件名（例如 'frame_0.jpg'）
            image_filename = base_json_filename.replace(".json", ".jpg")
            
            # 使用指定的图片文件夹路径来构建图片文件的完整路径
            image_file_path = os.path.join(image_folder_path, image_filename)

            image_width = None
            image_height = None
            try:
                if os.path.exists(image_file_path):
                    with Image.open(image_file_path) as img:
                        image_width, image_height = img.size
                else:
                    print(f"警告: 未找到对应的图片文件 {image_file_path}。将跳过图片信息添加。")
            except Exception as img_e:
                print(f"读取图片文件 {image_file_path} 时发生错误: {img_e}")

            # 添加图片信息到 AnyLabeling 格式数据中
            converted_data["imagePath"] = image_filename
            converted_data["imageData"] = None
            converted_data["imageHeight"] = image_height
            converted_data["imageWidth"] = image_width

            # 在该目录下创建一个新的输出子文件夹
            # 注意：这里 output_folder 仍是转换后的 JSON 的根输出目录，而不是 input_file_dir
            output_subfolder = os.path.join(output_folder, "anylabeling_output")
            os.makedirs(output_subfolder, exist_ok=True) # 确保新子文件夹存在

            output_file_path = os.path.join(output_subfolder, base_json_filename)

            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            print(f"成功转换文件: {file_path} -> {output_file_path}")
        except FileNotFoundError:
            print(f"错误: 未找到文件 {file_path}。")
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} 不是有效的 JSON 格式。")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")

    if os.path.isfile(input_path):
        process_file(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".json"):
                    json_file_path = os.path.join(root, file)
                    process_file(json_file_path)
    else:
        print(f"错误: 无效的输入路径 '{input_path}'。它必须是文件或文件夹。") 