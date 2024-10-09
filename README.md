# 1staged vs 2staged

## 개요
1staged-application이란 하나의 Object Detection 모델이 분류작업까지 마친 뒤 bbox를 그리는 앱이다. 2staged-application이란 Object Detection 모델은 단순 객체 인식을 하고, 뒤의 Classification 모델이 해당 boundary 내에서 분류작업을 하여 결과를 낸다. 두 개의 차이를 비교한다.

## 데이터셋 및 모델
1. 데이터셋
    1. Det_1
        - 차량을 Detection 한다.
    2. Det_8
        - 세부 차량을 Detection 한다.
    3. Cls_8
        - 차량에 대한 분류를 하는 모델이다.
2. 모델
    1. **Det_1model** : Det_1로 학습시킨 m 사이즈 모델 
    2. **Det_8model** : Det_8로 학습시킨 m 사이즈 모델
    3. **Cls_8model** : Cls_8로 학습시킨 s,m,l 모델

## 실험(비교) 내용
1. **Det_1model vs Det_8model**의 차량 검출 능력
    - Det_8model의 경우 모든 박스가 정답이 됨. 그에 따른 성능 비교
    - 비교 지표 : Recall, Precision
2. **Det_1model+Cls_8model vs Det_8model**의 모델 분류 능력
    - 연결한 어플리케이션과 단독 어플리케이션의 성능 비교
    - 비교 지표 : Topk-accuracy
3. **Det_1model+Cls_8model vs Det_8model**의 모델 감지 능력
    - 연결한 어플리케이션과 단독 어플리케이션의 성능 비교
    -  비교 지표 : Recall, Precision

### 규칙
1. 모든 모델은 동일한 조건의 데이터셋을 가진다.
    - 학습모델은 실험에 사용되는 test셋을 알고 있으면 안된다.
    - 따라서, 학습에 사용되는 데이터셋의 형태는 다를 수 있어도, 내용은 같아야한다.
    - Train:Val:Test = 0.8:0.1:0.1 로 Split된 Root 데이터셋을 기준으로 진행
2. 학습 시 동일한 환경에서 학습해야 한다. 
3. 평가 시 동일한 환경에서 평가를 해야한다.


### 해야할 순서
1. 데이터셋 Import : 규칙 1을 따른다.
2. 모델 Train : 규칙 2를 따른다.
3. 모델 평가 : 규칙 3를 따른다.
4. 어플리케이션 평가