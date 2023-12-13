from ultralytics import YOLO

try:
    model = YOLO("yolov8l-cls.pt", task="classify")
    model.train(cfg="/home/jyp/waffle/compare_1_2_staged/ultralytics/custom.yaml", data='/home/jyp/waffle/datasets/SOSDataset_v1.2.0/exports/YOLO')
except Exception as e:
    print(e)
    raise e

