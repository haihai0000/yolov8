# from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8s-pose.yaml')  # build a new model from YAML
# model = YOLO('yolov8s-pose.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s-pose.yaml').load('yolov8s-pose.pt')  # build from YAML and transfer weights

# # Train the model
# model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)

# ####自己的数据集训练
# from ultralytics import YOLO

# # 加载YOLOv8模型
# model = YOLO("yolov8s-pose.pt")  # 使用预训练的YOLOv8 Pose模型

# # 训练模型
# model.train(data="badminton_data.yaml", epochs=100, imgsz=640)  # 训练50个epoch，图像大小640


######## 普通余弦退火+adamw 训练1 #######
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO

# # 1. 加载官方预训练的 pose 模型，或者你自己的 runs/train/.../best.pt
# def main():
#     model = YOLO('yolov8s-pose.pt')

# # # 4. 开始训练：只有 pose‑head 会更新
#     model.train(
#     data="lin_shi.yaml",    # 你的数据配置
#     epochs=50,           # 训练轮数
#     # imgsz=640,           # 输入尺寸
#     batch=2,            # 根据显存调节
#     # lr0=1e-3,            # 初始学习率，通常比全量训练要小一点
#     device='0',          # 指定 GPU
#     workers=2,        # 这里设置为0
#     optimizer='AdamW',
#     cos_lr=True,  # 显式启用余弦退火
#     lr0=1e-4,     # 初始学习率
#     lrf=1e-5,     # 最终学习率（cos退火的最低点）
#     warmup_epochs=3,  # 学习率热身阶段
#     warmup_momentum=0.8,  # 初始动量
#     warmup_bias_lr=0.1,   # bias参数的学习率

# )
# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.set_start_method('spawn', force=True)  # Set spawn method for Windows
#     main()


# def main():
#     model = YOLO('yolov8s-pose.pt')
# # 自动找 lr
#     model.tune(data="lindan.yaml", epochs=1)
 
#     model.train(
#         data="lindan.yaml",
#         epochs=80,
#         patience=20,
#         batch=1,
#         imgsz=512,
#         lr0=5e-4,
#         lrf=0.01,
#         device='0',
#         workers=0,
#         save_period=5,
#         project="runs/pose_small",
#         name="exp1"
#     )

# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.set_start_method('spawn', force=True)
#     main()


###################  adam+线性衰减   #################
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     mp.freeze_support()
    
#     model = YOLO('yolov8s-pose.pt')  # 加载基础模型
    
#     # 使用最佳参数进行完整训练
#     model.train(
#         data="lindan.yaml",
#         epochs=100,  # 设置更多的训练轮数获得更好的结果
#         optimizer='Adam',  # 最佳优化器
#         lr0=0.003,  # 最佳初始学习率
#         lrf=0.01,  # 线性衰减到初始值的1%
#         cos_lr=False,  # 关闭余弦退火，使用线性衰减
#         weight_decay=0.0005,  # 最佳权重衰减
#         batch=16,  # 最佳批量大小
        
#         # 数据增强参数
#         hsv_h=0.015,
#         hsv_s=0.7,
#         hsv_v=0.4,
#         degrees=0.0,
#         translate=0.1,
#         scale=0.5,
        
#         # 其他参数
#         fliplr=0.5,  # 水平翻转概率
#         mosaic=0.8,  # 马赛克增强概率
#         workers=0,   # Windows系统下多线程设置
#         device=0,    # GPU设备编号
#         project="adam+linear",  # 项目名称
#         name="lindan_pose_adam+linear",  # 实验名称
#         exist_ok=True,  # 如果输出目录已存在则覆盖
#         patience=15,  # 早停设置
#         save=True,   # 保存模型
#         save_period=10  # 每10个epoch保存一次检查点
#     )


######################     训练2     ######################
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

# 1. 加载官方预训练的 pose 模型，或者自己的 runs/train/.../best.pt
def main():
    model = YOLO('yolov8s-pose.pt')

# # 4. 开始训练：只有 pose‑head 会更新
    model.train(
    data="lin_shi.yaml",    # 你的数据配置
    epochs=100,           # 训练轮数
    # imgsz=640,           # 输入尺寸
    batch=8,            # 根据显存调节
    # lr0=1e-3,            # 初始学习率，通常比全量训练要小一点
    device='0',          # 指定 GPU
    workers=2,        # 这里设置为0
    optimizer='AdamW',
    cos_lr=True,  # 显式启用余弦退火
    lr0=3e-4,     # 初始学习率
    lrf=1e-6,     # 最终学习率（cos退火的最低点）
    warmup_epochs=5,  # 学习率热身阶段
    warmup_momentum=0.8,  # 初始动量
    warmup_bias_lr=0.1,   # bias参数的学习率
    weight_decay=0.01,  # AdamW权重衰减参数
    patience=20,        # 早停耐心值

)
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # Set spawn method for Windows
    main()

 
######################  训练1+林丹大数据集   ######################
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO

# # 1. 加载官方预训练的 pose 模型，或者你自己的 runs/train/.../best.pt
# def main():
#     model = YOLO('yolov8s-pose.pt')

# # # # 4. 开始训练：只有 pose‑head 会更新
#     model.train(
#         data="lindan.yaml",
#         epochs=150,           # 增加轮数，小数据集需要更多轮次
#         batch=16,             # 适中的批次大小
#         device='0',
#         workers=4,
#         optimizer='AdamW',
#         cos_lr=True,
#         lr0=5e-4,            # 稍高的学习率
#         lrf=1e-5,            # 较高的最终学习率
#         warmup_epochs=3,      # 较短的预热
#         warmup_momentum=0.8,
#         warmup_bias_lr=0.1,
#         weight_decay=0.005,   # 减小权重衰减
#         patience=25,          # 增加耐心值
#     )
# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.set_start_method('spawn', force=True)  # Set spawn method for Windows
#     main()
