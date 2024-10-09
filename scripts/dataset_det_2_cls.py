import os
import cv2
from tqdm import tqdm
from waffle_utils.file import io

def traverse_files_in_folder(folder_path):
    """Yolo 전용, 폴더 내 파일에서 확장자 제외한 상대경로 전체 리스트를 구하는 함수"""
    # 폴더 내의 모든 파일 순회
    relative_file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            relative_file_path = os.path.relpath(os.path.join(root, file), folder_path)
            result_path = relative_file_path.split(".")[0]
            relative_file_list.append(result_path)
    return relative_file_list
            
def parse_cate_name(data_yaml_path: str):
    """ data.yaml의 path를 받아 아이디: 카테고리 이름 dict를 반환한다."""
    data_dict = io.load_yaml(data_yaml_path)
    return data_dict["names"]

###############################################################################################
def det_2_cls(det_yolo_dir,result_yolo_dir = None):
    """
    객체 탐지 목적으로 만들어진 YOLO 폴더를 기반으로, image 및 label 대칭으로 Crop을 진행한후 classification 학습을 위한 Yolo데이터셋으로 제작하는 함수이다.
    train, val, test폴더가 들어있다고 가정한다.
    data.yaml을 통해 카테고리 이름을 parse 한다.
    """
    # 기존 욜로
    train_dir = f"{det_yolo_dir}/train"
    val_dir = f"{det_yolo_dir}/val"
    test_dir = f"{det_yolo_dir}/test"
    data_file = f"{det_yolo_dir}/data.yaml"
    target_dir = [train_dir, val_dir, test_dir]
    cate_dict = parse_cate_name(data_file)
    
    # 결과가 될 파일
    result_train_dir = f"{result_yolo_dir}/train"
    result_val_dir = f"{result_yolo_dir}/val"
    result_test_dir = f"{result_yolo_dir}/test"
    result_dir = [result_train_dir, result_val_dir, result_test_dir]
    
    match_dict ={
        train_dir : result_train_dir,
        val_dir : result_val_dir,
        test_dir : result_test_dir
    }
    
    # yolo 디렉터리 구조 생성
    for dir in result_dir:
        os.makedirs(dir, exist_ok=True)

    # train, val, test 폴더 순회
    for dir in target_dir:
        get_file_list = traverse_files_in_folder(dir + "/images")       # 폴더 내의 파일들 정보 획득
        for cate in cate_dict.values():                                 # 카테고리 정보에 따른 폴더 구조 생성
            cate_dir = match_dict[dir] + "/images/" + cate
            os.makedirs(cate_dir, exist_ok = True)
        for file in tqdm(get_file_list, desc = "Processing Images"):                                      # 파일 각각 수행
            annotation_list = []                                        # 어노테이션 정보를 저장
            txt_file = dir + "/labels/" + file + ".txt"
            image_file = dir + "/images/" + file + ".jpg"   # or png
            image = cv2.imread(image_file)
            height, width, _ = image.shape
            
            with open(txt_file, 'r') as f:               # get annotations in txt file
                for line in f:                           # 라인별 하나씩 박스 정보를 가짐
                    ann_ = {"category_id": line.split(" ")[0],
                            "bbox": [(float(line.split(" ")[1])-float(line.split(" ")[3])/2)*width, (float(line.split(" ")[2])-float(line.split(" ")[4])/2)*height,
                                     (float(line.split(" ")[1])+float(line.split(" ")[3])/2)*width, (float(line.split(" ")[2])+float(line.split(" ")[4])/2)*height]
                    }
                    
                    
                    annotation_list.append(ann_)            # 어노테이션 정보를 가진 리스트 획득
            
            id = 0
            for ann in annotation_list:
                bbox = list(map(int, ann["bbox"]))  #int화 된 bbox 리스트
                cropped_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                dst = match_dict[dir] + "/images/" + cate_dict[int(ann["category_id"])] + "/"+ str(id) +"_"+ file.split("/")[-1] + ".jpg"
                cv2.imwrite(dst,cropped_img)
                id += 1
                

if __name__ == "__main__":
    yolo_dir = "/home/jyp/waffle/datasets/Ison_Det_8/exports/ULTRALYTICS"
    result_dir = "/home/jyp/waffle/datasets/Ison_Cls_8/exports/ULTRALYTICS"
    
    det_2_cls(yolo_dir, result_dir)