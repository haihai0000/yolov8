import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import multiprocessing

def main():
    # Load a model
    model = YOLO("yolov8s-pose.pt")  # load an official model
    model = YOLO("runs/pose/train89/weights/best.pt")  # load a custom model

    # 完整验证测试集（需要测试集标签）
    test_metrics = model.val(
        data='lin_shi.yaml',
        split='test',     # 明确指定使用测试集
        batch=4,         # 根据显存调整
        conf=0.25,
        iou=0.6,
        save_json=True,   # 生成JSON格式结果
        save_hybrid=True  # 保存混合标签结果
    )

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
    metrics.pose.map  # map50-95(P)
    metrics.pose.map50  # map50(P)
    metrics.pose.map75  # map75(P)
    #metrics.pse.maps  # a list contains map50-95(P) of each category}

if __name__ == '__main__':
    # Windows专用设置
    multiprocessing.freeze_support()  # 添加冻结支持
    multiprocessing.set_start_method('spawn', force=True)
    main()


# ######使用未训练过的模型进行推理#####
# import os
# import multiprocessing
# from ultralytics import YOLO

# def run_inference_and_evaluate(
#     model_path: str,
#     data_yaml: str,
#     images_dir: str,
#     results_dir: str = "runs/pose/inference",
#     conf: float = 0.25,
#     iou: float = 0.6,
#     batch_size: int = 4
# ):
#     """
#     1) 加载模型（官方或自训练）。
#     2) 对 images_dir 中的所有图片做推理，将结果保存到 results_dir。
#     3) 使用 model.val()（依赖之前 val 时指定的 data_yaml）来计算 box 和 pose 的 mAP 指标。
#     """

#     # 避免 OpenMP 重复加载报错
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#     # 1. 加载模型
#     model = YOLO(model_path)

#     # 2. 推理（predict）
#     #    - source: 图片文件夹或单张图片路径
#     #    - save:   将可视化结果（带 keypoints、bounding boxes）保存到 runs/pose/inference
#     #    - conf:   置信度阈值
#     #    - iou:    NMS 的 IOU 阈值
#     #    - batch:  批量大小，按显存调整
#     results = model.predict(
#         source=images_dir,
#         save=True,
#         save_dir=results_dir,
#         conf=conf,
#         iou=iou,
#         batch=batch_size,
#         device=0  # 如果有多张卡可改为 'cuda:0','cuda:1' 等
#     )
#     print(f"Inference results saved to {results_dir}")

#     # 3. 评估（validate）
#     #    注意：第一次调用 model.val(...) 时请带上 data_yaml 和 split='test'
#     #          这样才能让模型记住数据集配置，后面直接 model.val() 即可复用。
#     test_metrics = model.val(
#         data='lindan.yaml',
#         split='test',
#         batch=batch_size,
#         conf=conf,
#         iou=iou,
#         save_json=True,
#         save_hybrid=True
#     )
#     print("==== Test set evaluation ====")
#     print(f"Box mAP 50–95 : {test_metrics.box.map:.4f}")
#     print(f"Box mAP50      : {test_metrics.box.map50:.4f}")
#     print(f"Box mAP75      : {test_metrics.box.map75:.4f}")
#     print(f"Per-class box mAPs: {test_metrics.box.maps}")

#     print(f"Pose mAP 50–95 : {test_metrics.pose.map:.4f}")
#     print(f"Pose mAP50     : {test_metrics.pose.map50:.4f}")
#     print(f"Pose mAP75     : {test_metrics.pose.map75:.4f}")
#     # 如果想要每个类别的 pose mAP 列表，可以使用：
#     # print(f"Per-class pose mAPs: {test_metrics.pose.maps}")

#     # 4. 复用上面记住的设置，快速重复调用
#     metrics = model.val()  # 直接复用 data_yaml、split、conf、iou 等设置
#     print("==== Re-used settings evaluation ====")
#     print(f"Box mAP 50–95 : {metrics.box.map:.4f}")
#     print(f"Pose mAP 50–95: {metrics.pose.map:.4f}")

# if __name__ == '__main__':
#     # Windows 多进程支持
#     multiprocessing.freeze_support()
#     multiprocessing.set_start_method('spawn', force=True)

#     # 示例参数，请根据实际情况修改：
#     model_path   = "yolov8s-pose.pt"  # 或 "yolov8s-pose.pt"
#     data_yaml    = "lindan.yaml"                       # 包含 train/val/test 划分的 dataset 配置
#     images_dir   = "Z:/BadData/Benchmark/Benchmark_pose/test/shike_test/images"          # 推理图片所在目录
#     results_dir  = "runs/pose/inference"

#     run_inference_and_evaluate(
#         model_path=model_path,
#         data_yaml=data_yaml,
#         images_dir=images_dir,
#         results_dir=results_dir,
#         conf=0.25,
#         iou=0.6,
#         batch_size=4
#     )
