
######普通寻找超参数##############
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     # Windows系统设置
#     mp.set_start_method('spawn', force=True)
    
#     # 加载预训练模型
#     model = YOLO('yolov8s-pose.pt')
    
#     # 超参数搜索（针对小数据集优化的搜索空间）
#     model.tune(
#         data="lindan.yaml",
#         space={
#             # 学习率相关（小数据集需要较小学习率）
#             'lr0': [0.0001, 0.0003, 0.0005, 0.001],
#             'lrf': [0.01, 0.05, 0.1],
            
#             # 正则化参数（防止过拟合）
#             'weight_decay': [0.0003, 0.0005, 0.001],
#             'dropout': [0.0, 0.1, 0.2],
            
#             # 数据增强（小数据集需要更强的增强）
#             'hsv_h': [0.01, 0.015, 0.02],
#             'hsv_s': [0.5, 0.7, 0.9],
#             'hsv_v': [0.3, 0.4, 0.5],
#             'degrees': [5.0, 10.0, 15.0],
#             'translate': [0.05, 0.1, 0.15],
#             'scale': [0.3, 0.5, 0.7],
#             'fliplr': [0.0, 0.5], # 在0到0.5之间随机选择
#             'mosaic': [0.5, 0.8, 1.0],
            
#             # 批量大小（小数据集适中即可）
#             'batch': [2, 4],
#         },
#         epochs=30,          # 每组参数训练30轮足够评估
#         iterations=8,       # 尝试8种参数组合
#         workers=0,          # Windows系统设置
#         device=0,
#         plots=True,         # 生成分析图表
#         save=True,          # 保存最佳模型
#         seed=42,            # 固定随机种子确保可比性
#         val=True            # 在验证集上评估
#     )

#########结合余弦退火学习器和Adamw优化器来找参数##############
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     # 解决Windows多进程问题
#     mp.freeze_support()
    
#     # 加载预训练模型
#     model = YOLO('yolov8s-pose.pt')
    
#     # 启动超参数调优，指定数据集、epochs和搜索次数
#     results = model.tune(
#         data="lindan.yaml",
#         epochs=30,
#         iterations=10,  # 会自动尝试10种不同的参数组合
#         optimizer='AdamW',  # 使用AdamW优化器
#         lr0=0.001,  # 基准学习率，tune方法会自动变化这个值
#         lrf=0.05,  # 基准最终学习率系数
#         momentum=0.9,  # 基准动量值
#         weight_decay=0.001,  # 基准权重衰减值
#         warmup_epochs=2.0,  # 基准预热周期
#         warmup_momentum=0.6,  # 基准预热动量
#         workers=0,
#         batch=4
#     )

#############  1、不同优化器测试   ##############
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     mp.freeze_support()
    
#     model = YOLO('yolov8s-pose.pt')
    
#     # 测试不同优化器
#     optimizers = [ 'Adam', 'AdamW', 'RMSProp']
    
#     for opt in optimizers:
#         print(f"测试优化器: {opt}")
        
#         # 为每个优化器定义适当的搜索空间
#         if opt in [ 'RMSProp']:
#             space = {
#                 'lr0': [0.0005, 0.001, 0.002],
#                 'weight_decay': [0.0001, 0.0005, 0.001],
#                 'momentum': [0.8, 0.9, 0.95],
#             }
#         else:  # Adam, AdamW不需要momentum参数
#             space = {
#                 'lr0': [0.0005, 0.001, 0.002],
#                 'weight_decay': [0.0001, 0.0005, 0.001],
#             }
        
#         model.tune(
#             data="lindan.yaml",
#             epochs=20,
#             iterations=5,
#             optimizer=opt,
#             space=space,
#             workers=0,
#             batch=4,
#             name=f"tune_{opt}"
#         )


#############  2、学习率调度器与优化器组合测试   ##############
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     mp.freeze_support()
    
#     model = YOLO('yolov8s-pose.pt')
    
#     # 测试最佳的优化器和学习率调度器组合
#     # 基于方案一中表现最好的优化器(假设是AdamW)
    
#     # 余弦退火学习率
#     model.tune(
#         data="lindan.yaml",
#         epochs=30,
#         iterations=8,
#         optimizer='AdamW',
#         space={
#             'lr0': [0.0005, 0.001, 0.002],
#             'lrf': [0.01, 0.05, 0.1],
#             'weight_decay': [0.0001, 0.0005, 0.001, 0.005],
#             'warmup_epochs': [1.0, 2.0, 3.0],
#             'warmup_momentum': [0.5, 0.6, 0.8],
#         },
#         workers=0,
#         batch=4,
#         name="tune_cosine_lr"
#     )
    
#     # 一阶段学习率
#     model.train(
#         data="lindan.yaml",
#         epochs=30,
#         optimizer='AdamW',
#         lr0=0.001,
#         lrf=1.0,  # 保持学习率不变
#         weight_decay=0.001,
#         workers=0,
#         batch=4,
#         name="constant_lr"
#     )
    
#     # 线性学习率衰减
#     model.train(
#         data="lindan.yaml",
#         epochs=30,
#         optimizer='AdamW',
#         lr0=0.001,
#         lrf=0.01,  # 线性衰减到初始值的1%
#         weight_decay=0.001,
#         workers=0,
#         batch=4,
#         name="linear_lr"
#     )
###########adam+不同的学习率器###########
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     mp.freeze_support()
    
#     model = YOLO('yolov8s-pose.pt')
    
#     # 1. Adam + 余弦退火学习率
#     model.tune(
#         data="lindan.yaml",
#         epochs=30,
#         iterations=8,
#         optimizer='Adam',
#         weight_decay=0.0005,  # 固定参数
#         space={
#             'lr0': [0.0005, 0.002],
#             'lrf': [0.01, 0.1],
#             'warmup_epochs': [1.0, 3.0],
#             'warmup_momentum': [0.5, 0.8],
#             'batch': [4, 16],
#         },
#         workers=0,
#         device=0,
#         name="tune_adam_cosine"
#     )
    
#     # 2. Adam + 恒定学习率
#     model.train(
#         data="lindan.yaml",
#         epochs=30,
#         optimizer='Adam',
#         lr0=0.001,  # 使用之前找到的最佳学习率
#         lrf=1.0,  # 保持学习率不变
#         weight_decay=0.0005,
#         workers=0,
#         batch=4,
#         name="adam_constant_lr"
#     )
    
#     # 3. Adam + 线性学习率衰减
#     model.train(
#         data="lindan.yaml",
#         epochs=30,
#         optimizer='Adam',
#         lr0=0.001,
#         lrf=0.01,  # 线性衰减到初始值的1%
#         weight_decay=0.0005,
#         workers=0,
#         batch=4,
#         name="adam_linear_lr"
#     )
    
#     # 4. Adam + 阶梯式学习率衰减
#     # 通过设置step参数实现
#     model.train(
#         data="lindan.yaml",
#         epochs=30,
#         optimizer='Adam',
#         lr0=0.001,
#         lrf=0.1,  # 最终学习率为初始值的10%
#         cos_lr=False,  # 关闭余弦退火
#         weight_decay=0.0005,
#         workers=0,
#         batch=4,
#         name="adam_step_lr"
#     )

#############  3、精细超参数搜索   ##############
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from ultralytics import YOLO
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     mp.freeze_support()
    
#     model = YOLO('yolov8s-pose.pt')
    
#     # 基于前两个方案的结果，进行更精细的搜索
#     model.tune(
#         data="lindan.yaml",
#         epochs=50,  # 增加训练轮数以获得更稳定的结果
#         iterations=15,
#         optimizer='AdamW',
#         space={
#             # 优化器相关参数
#             'lr0': [0.0008, 0.001, 0.0012],  # 更精细的学习率范围
#             'weight_decay': [0.0008, 0.001, 0.0012],
            
#             # 学习率调度相关
#             'lrf': [0.03, 0.05, 0.07],  # 余弦退火最终学习率因子
#             'warmup_epochs': [1.5, 2.0, 2.5],
#             'warmup_momentum': [0.65, 0.7, 0.75],
            
#             # 数据增强相关参数
#             'hsv_h': [0.01, 0.015, 0.02],
#             'hsv_s': [0.6, 0.7, 0.8],
#             'hsv_v': [0.4, 0.5, 0.6],
#             'degrees': [0.0, 5.0, 10.0],
#             'translate': [0.1, 0.15, 0.2],
#             'scale': [0.4, 0.5, 0.6],
#             'fliplr': [0.5],
#             'mosaic': [0.7, 0.8, 0.9],
            
#             # 姿态估计特定参数
#             'kobj_scale': [1.0, 1.2, 1.5],  # 关键点置信度损失权重
#             'perspective': [0.0, 0.0005, 0.001],  # 透视变换
            
#             # 其他训练参数
#             'dropout': [0.0, 0.1],
#             'box': [7.5, 10.0, 12.5],  # 边界框损失权重
#             'cls': [0.3, 0.5, 0.7],  # 分类损失权重
#         },
#         workers=0,
#         batch=4,
#         device=0,
#         plots=True,
#         save=True,
#         seed=42,
#         val=True,
#         name="final_tune"
#     )

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.freeze_support()
    
    model = YOLO('yolov8s-pose.pt')

    # 进行最终的精细超参数搜索
    model.tune(
        data="lindan.yaml",
        epochs=50,
        iterations=15,
        optimizer='Adam',
        lrf=0.01,  # 线性衰减参数作为固定值
        cos_lr=False,  # 关闭余弦退火
        fliplr=0.5,  # 固定参数直接传递
        mosaic=0.8,  # 固定参数直接传递
        space={
            # 每个参数都使用[最小值, 最大值]的格式
            'lr0': [0.001, 0.003],
            'weight_decay': [0.0003, 0.0007],
            'batch': [8, 32],
            'hsv_h': [0.01, 0.02],
            'hsv_s': [0.6, 0.8],
            'hsv_v': [0.4, 0.6],
            'degrees': [0.0, 10.0],
            'translate': [0.1, 0.2],
            'scale': [0.4, 0.6],
        },
        workers=0,
        device=0,
        plots=True,
        save=True,
        seed=42,
        val=True,
        name="adam_linear_final3"
    )
