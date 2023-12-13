import json
import shutil
import os
import copy

import cv2
from tqdm import tqdm
from waffle_hub.hub import Hub
from waffle_hub.dataset import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from waffle_utils.file import io
from ultralytics import YOLO

# step 1.1) waffle_hub의 inference를 이용한 od 결과 추출
def waffle_od_inference(od_hub, dataset_source):
    if type(od_hub) == str:
        od_hub = Hub.load(name = od_hub)
    return od_hub.inference(dataset_source).predictions
    #io.save_json(result, result_json)

# 이미지 이름에 따른 이미지 아이디를 얻는 함수
def get_image_info(gt):
    img_dict = {}
    if type(gt) == str:
        gt = io.load_json(gt)
    for image in gt["images"]:
        img_dict[image["file_name"]] = image["id"]
    return img_dict

# 데이터셋중 set_file에 있는 것만 가져오는 함수
def split_gt(gt, set_file):
    if type(gt) == str:
        gt = io.load_json(gt)
    
    set_json = io.load_json(set_file)
    
    new_gt = {}
    new_gt_img_list = []
    new_gt_ann_list = []
    
    for img in gt["images"]:
        if img["id"] not in set_json:
            continue
        else:
            new_img_ = img
        new_gt_img_list.append(new_img_)
    
    for ann in gt["annotations"]:
        if ann["image_id"] not in set_json:
            continue
        else:
            new_ann_ = ann
        new_gt_ann_list.append(new_ann_)
        
    new_gt = {
        "images": new_gt_img_list,
        "annotations": new_gt_ann_list,
        "categories": gt["categories"]
    }
    
    return new_gt
    
    
# step 1.2) 1.1의 결과를 coco val list에 맞게 변경
def waffle_inference_to_coco_list(waffle_inference, image_dict):
    result_list = list()
    anns_id_count = 0
    if type(waffle_inference) == str:
        waffle_inference = io.load_json(waffle_inference)
    
    for infer in waffle_inference:
        if list(infer.values())[0] == []:
            continue
        else:
            for anns in list(infer.values())[0]:
                anns_id = int(anns_id_count)
                image_id = int(image_dict[list(infer.keys())[0]])
                category_id = int(anns["category_id"])
                area = float(anns['bbox'][2]*anns['bbox'][3])
                bbox = anns['bbox']
                score = anns['score']
                
                anns_dict = {
                    "id": anns_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "score": score
                }
                result_list.append(anns_dict)
                anns_id_count += 1
    return result_list

# step 2. workspace에 이미지들 crop
def infer_cropper(infer_json, image_dict, input_img_folder,result_folder) :
    if type(infer_json) == str:
        infer_json = io.load_json(infer_json)
    
    # image_dict 리버스하기
    swapped_dict = {v: k for k, v in image_dict.items()}
    
    for i in tqdm(infer_json, desc = "Processing Images") :
        image_id = i["image_id"]
        ann_id = i["id"]
        img_path = f"{input_img_folder}/{swapped_dict[image_id]}"
        result_path = f"{result_folder}/{ann_id}_{image_id}.jpg"
        
        bbox = list(map(int, i["bbox"]))
        image_load = cv2.imread(img_path)
        cropped_img = image_load[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        cv2.imwrite(result_path,cropped_img)

# 와플로 하는거, 잘안됨
# # step 3 Cls 모델로 crop한 이미지 평가
# def waffle_cls_inference(cls_hub, dataset_source):
#     if cls_hub == str:
#         cls_hub = Hub.load(name = cls_hub)
#     infer_list = cls_hub.inference(source=dataset_source, iou_threshold=0.6).predictions
    
#     cls_infer_dict = {}
#     for ann in infer_list:
#         for image_id, categories in ann.items():
#             max_score_category = max(categories, key=lambda x: x["score"])
#             cls_infer_dict[image_id] = max_score_category["category_id"]

#     return cls_infer_dict



# step 3 YOlo로 Cls 모델로 crop한 이미지 평가
def yolo_cls_predict(cls_hub, dataset_source):
    cls_infer_dict = {}
    
    cls_file = cls_hub.best_ckpt_file
    model = YOLO(cls_file)
    results = model.predict(source = dataset_source)
    for result in results:
        top1 = result.probs.top1
        jpg_name = result.path.split("/")[-1]
        cls_infer_dict[jpg_name] = top1
        
    return cls_infer_dict

# step 4. coco 예측지 (dt.json)에서 카테고리 변경
def change_dt(step_1_2_list, step_3_dict):
    if type(step_1_2_list) == str:
        step_1_2_list = io.load_json(step_1_2_list)
    if type(step_3_dict) == str:
        step_3_dict = io.load_json(step_3_dict)
        
    new_result = []
    for ann in step_1_2_list:
        image_id_in_1_2 = str(ann["id"]) + "_" + str(ann["image_id"])
        cls_img = f"{image_id_in_1_2}.jpg"
        new_result_ = {
            "id": ann["id"],
            "image_id": ann["image_id"],
            "category_id": step_3_dict[cls_img],
            "area": ann["area"],
            "bbox": ann["bbox"],
            "iscrowd": 0,
            "score": ann["score"]
        }

        new_result.append(new_result_)
    
    return new_result

# step 5. coco 답지와 예측지를 통한 mAP 측정기
def eval_gt_dt(gt_json, dt_json):
        
    annType = 'bbox'
    cocoGt=COCO(gt_json)
    
    cocoDt=cocoGt.loadRes(dt_json)
    imgIds=sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
  
# 와플로할떄  
# # 클래스 매칭 하드코딩
# def dt_match(det_dt):
#     if type(det_dt) == str:
#         det_dt = io.load_json(det_dt)
#     copy_dt = copy.deepcopy(det_dt)
#     for ann in copy_dt:
#         if ann["category_id"] == 8:
#             ann["category_id"] = 1
#         elif ann["category_id"] == 4:
#             ann["category_id"] = 2
#         elif ann["category_id"] == 2:
#             ann["category_id"] = 3
#         elif ann["category_id"] == 5:
#             ann["category_id"] = 4
#         elif ann["category_id"] == 7:
#             ann["category_id"] = 5
#         elif ann["category_id"] == 1:
#             ann["category_id"] = 6
#         elif ann["category_id"] == 3:
#             ann["category_id"] = 7
#         elif ann["category_id"] == 6:
#             ann["category_id"] = 8
#         elif ann["category_id"] == 9:
#             ann["category_id"] = 9
#     return copy_dt

# yolo predict로 할떄
def dt_match(det_dt):
    if type(det_dt) == str:
        det_dt = io.load_json(det_dt)
    copy_dt = copy.deepcopy(det_dt)
    for ann in copy_dt:
        if ann["category_id"] == 0:
            ann["category_id"] = 6
        elif ann["category_id"] == 1:
            ann["category_id"] = 3
        elif ann["category_id"] == 2:
            ann["category_id"] = 7
        elif ann["category_id"] == 3:
            ann["category_id"] = 2
        elif ann["category_id"] == 4:
            ann["category_id"] = 4
        elif ann["category_id"] == 5:
            ann["category_id"] = 8
        elif ann["category_id"] == 6:
            ann["category_id"] = 5
        elif ann["category_id"] == 7:
            ann["category_id"] = 1
        elif ann["category_id"] == 8:
            ann["category_id"] = 9
    return copy_dt
    
if __name__== "__main__":
    # fix here
    gt = '/home/jyp/waffle/datasets/IsonDataset_coco/coco_detector_9cate_without_people.json'         # 답안지
    od_hub = Hub.load(name = "IsonDet_2class_l_v1.0.0")
    cls_hub = Hub.load(name = "IsonCls_9class_l_v1.0.0")
    od_dataset_source = "/home/jyp/waffle/datasets/IsonDataset_Det_v1.0.0/exports/ULTRALYTICS/test/images"
    split_json = "/home/jyp/waffle/datasets/IsonDataset_Det_v1.0.0/sets/test.json"
    gt_path = "./2_gt.json"
    dt_path = "./2_dt.json"
    
    new_gt = split_gt(gt,split_json)    # 답안지를 해당 split된 set만 적용하도록함
    workspace = "./workspace"           # crop된 이미지 저장할 장소
    io.make_directory(workspace)

    
    step_1_1 = waffle_od_inference(od_hub, od_dataset_source)
    io.save_json(step_1_1,"./step_1_1.json")
    step_1_2 = waffle_inference_to_coco_list(step_1_1, get_image_info(new_gt))
    io.save_json(step_1_2,"./step_1_2.json")
    step_2 = infer_cropper(step_1_2, get_image_info(new_gt), od_dataset_source, workspace)
    step_3 = yolo_cls_predict(cls_hub = cls_hub, dataset_source=workspace)
    io.save_json(step_3, './step_3.json')
    step_1_2 = "./step_1_2.json"
    dt = change_dt(step_1_2, step_3)    # step_4
    io.save_json(step_3,"./step_3.json")
    io.save_json(dt, dt_path)
    io.save_json(new_gt, gt_path)
    
    dt_hardcode = dt_match(dt_path)          # cls 카테고리가 이상해서 매칭하는 하드코딩이 필요함..;;
    io.save_json(dt_hardcode, dt_path)
    eval_gt_dt(gt_path, dt_path)                   # step_5
    
    io.remove_file(gt_path)
    io.remove_file(dt_path)
    io.remove_file("./step_3.json")
    io.remove_directory(workspace)
    
