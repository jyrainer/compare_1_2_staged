from tqdm import tqdm
from waffle_hub.hub import Hub
from waffle_hub.dataset import Dataset



# dataset = Dataset.from_coco(
#     name = "Ison_Det_1",
#     task = "OBJECT_DETECTION",
#     coco_file="/home/jyp/waffle/datasets/IsonDataset_coco/coco_1.json",
#     coco_root_dir = "/home/jyp/waffle/datasets/IsonDataset_coco/images"
# )

dataset = Dataset.load( name = "Ison_Det_1")

#_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
#print(len(train_ids), len(val_ids), len(test_ids), len(unlabeled_ids))
dataset.export(
    "yolo"
)