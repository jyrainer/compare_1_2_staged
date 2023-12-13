import copy

import cv2
from tqdm import tqdm
from waffle_hub.hub import Hub
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from waffle_utils.file import io

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
    
# 클래스 매칭 하드코딩
def dt_match_1(det_dt):
    if type(det_dt) == str:
        det_dt = io.load_json(det_dt)
    new_list = []
    for ann in det_dt:
        if ann["category_id"] == 1:
            pass
        else:
            content_ = {
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": ann["category_id"] - 1,
                "area": ann["area"],
                "bbox": ann["bbox"],
                "iscrowd": 0,
                "score": ann["score"]
            }
            new_list.append(content_)
        
    return new_list
    
if __name__== "__main__":
    # fix here
    gt = '/home/jyp/waffle/datasets/IsonDataset_coco/coco_detector_9cate_without_people.json'         # 답안지
    od_hub = Hub.load(name = "IsonDet_10class_s_v1.0.0")
    od_dataset_source = "/home/jyp/waffle/datasets/IsonDataset_Det_v1.0.0/exports/ULTRALYTICS/test/images"
    
    split_json = "/home/jyp/waffle/datasets/IsonDataset_Det_v1.0.0/sets/test.json"

    new_gt = split_gt(gt,split_json)    # 답안지를 해당 split된 set만 적용하도록함

    gt_path = "./1_gt.json"
    dt_path = "./1_dt.json"
    
    step_1_1 = waffle_od_inference(od_hub, od_dataset_source)
    io.save_json(step_1_1,"./step_1_1_1.json")
    step_1_1 = './step_1_1_1.json'
    step_1_2 = waffle_inference_to_coco_list(step_1_1, get_image_info(new_gt))
    io.save_json(step_1_2,"./step_1_2_1.json")
    io.save_json(step_1_2, dt_path)
    io.save_json(new_gt, gt_path)
    dt_hardcode = dt_match_1(dt_path)          # -1씩
    io.save_json(dt_hardcode, dt_path)
    eval_gt_dt(gt_path, dt_path)                   # step_5