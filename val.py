# from ultralytics import YOLO

# # Load a model
# #model = YOLO('yolov8n-pose.pt')  # load an official model
# model = YOLO('runs/pose/train/weights/best.pt')  # load a custom model我自己的模型

# # Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category

# # val.py解决多进程的问题
# from ultralytics import YOLO
# import torch.multiprocessing as mp
# import cv2

# def main():
#     # 加载训练好的模型
#     model = YOLO('runs/pose/train/weights/best.pt')
    
#     # 测试多人图像
#     results = model.predict(
#         source=r"C:/Users/Administrator/Desktop/1.png",
#         conf=0.25,     # 设置适当的置信度阈值
#         iou=0.45,      # 设置适当的IOU阈值
#         show=True
#     )
    
#     # 打印检测到的目标数量
#     print(f"检测到 {len(results[0].boxes)} 个人物")
#     print(f"检测到 {len(results[0].keypoints)} 组关键点")
    
#     # 可视化结果
#     img = cv2.imread(r"C:/Users/Administrator/Desktop/1.png")
#     res_plotted = results[0].plot()
#     cv2.imwrite('results.jpg', res_plotted)
    
#     # 验证指标
#     metrics = model.val()
#     print(f"检测mAP: {metrics.box.map50:.4f}")
#     print(f"姿态mAP: {metrics.pose.map50:.4f}")

# if __name__ == '__main__':
#     mp.freeze_support()
#     main()


# from ultralytics import YOLO
# import torch.multiprocessing as mp
# import os

# def main():
#     # 设置环境变量解决Windows多进程问题
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
#     # 加载模型
#     model = YOLO('runs/pose/train/weights/best.pt')  # 模型路径，如'runs/pose/train/weights/best.pt'
    
#     # 设置验证参数
#     metrics = model.val(
#         data="badminton_data.yaml",
#         batch=8,        # 减小批量大小以避免内存问题
#         imgsz=640,      # 设置与训练相同的图像大小
#         workers=0,      # Windows平台关键设置：使用0个工作进程
#         device=0,       # 使用GPU 0
#         verbose=True    # 详细输出
#     )
    
#     # 打印验证结果
#     print("验证完成！")
#     print(f"指标结果: {metrics}")

# if __name__ == '__main__':
#     # 解决Windows多进程问题的关键代码
#     mp.set_start_method('spawn', force=True)
#     mp.freeze_support()  # 必须的，解决Windows上的多进程问题
#     main()