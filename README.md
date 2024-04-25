[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/nDCOQnZo)

# Receipt Text Detection
## OCR 2조

| ![우승현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![송현지](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이재민](https://avatars.githubusercontent.com/u/156163982?v=4) | ![배창현](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [우승현](https://github.com/fromwsh)             |            [송현지](https://github.com/UpstageAILab)             |            [이재민](https://github.com/UpstageAILab)             |            [배창현](https://github.com/UpstageAILab)             |
|                            팀장                             |                            팀원                             |                            팀원                             |                            팀원                             |


### **1. 경진대회 개요**

- Receipt Text Detection

1. 경진대회 구현 내용, 컨셉, 교육 내용과의 관련성 등:
 이번 영수증 글자 검출 대회에서는 영수증에서 문자를 정확히 검출하는 것을 목표로 합니다. 이 때 AI 기반 training 학습을 통한 OCR 기술(광학 문자 인식)에 초점을 맞추고 있습니다. 이는 실생활에서 매우 유용한 기술로, 개인 및 기업의 재무 관리, 소비 패턴 분석, 자동 회계 처리 등 다양한 분야에 활용될 수 있는 것이며 현재에도 이미 사용이 되고 있습니다. 결과물을 제출하고 점수를 확인해가며 코드를 계속해서 수정하고 최적화 해가면서 높은 검출 성능을 낼 수 있는 방법을 계속해서 찾아가는 과정입니다. 이러한 경진대회는 업무 자동화 및 데이터 활용에 대한 중요성을 강조하며, 참가자들에게는 최신 AI 기술을 활용하여 현실적인 문제에 대한 해결책을 모색하고 구현하는 능력을 기르는 것이 목표입니다.

2. 활용 장비 및 재료(개발 환경, 협업 tool 등):
 기본적으로 우리가 늘 사용하는 Python과 딥러닝 프레임워크인 TensorFlow 또는 PyTorch를 사용하여 모델을 구현하고 학습시킬 것입니다. 업스테이지에서 제공해주는 코드 패키지에서도 이미 PyTorch를 통해 코드가 구현되어 있기 때문에, 이것을 기반으로 여러 조건들을 수정해가면서 H-Mean, Precision, Recall 값이 가장 높아지는 방향으로 OCR을 계속해서 진행하였습니다.  또한, 이미지 처리를 위해 OpenCV와 같은 라이브러리도 활용될 수 있습니다.

3. 경진대회 구조 및 사용 데이터셋의 구조도(연관도):
 이 대회의 구조는 주어진 학습용 데이터셋을 활용하여 모델을 학습하고, 평가용 데이터셋을 활용하여 모델의 성능을 검증하는 과정으로 이루어집니다. 학습용 데이터셋은 약 3,200여장의 영수증 이미지와 이에 대한 정답 데이터가 제공되며, 평가용 데이터셋은 총 413장의 영수증 이미지로 구성되어 있습니다. 또한, 데이터셋의 구조는 각 영수증 이미지마다 평균 100여개의 Text Region이 있으며, 이는 Polygon 좌표로 Labeling되어 있습니다. 이러한 구조는 모델을 학습할 때 영수증 이미지 내의 글자 영역을 정확하게 검출하기 위한 ground truth로 활용됩니다. 이러한 데이터셋 구조는 모델을 학습하고 검증하는 데 있어서 중요한 역할을 합니다. 데이터셋의 다양성과 실제 상황을 반영한 구조는 모델의 일반화 성능을 높이는 데 기여하며, 모델의 강인성과 효율성을 평가하는 데 중요한 지표로 활용될 것입니다.

※ 사용 데이터셋의 구조도(연관도)
├─── images
│   ├── train
│   │   ├── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   ├── ...
│   │   └── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   ├── val
│   │   ├── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   ├── ...
│   │   └── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   └── test
│   │   ├── drp.en_ko.in_house.selectstar_NNNNNN.jpg
│   │   ├── ...
│   │   └── drp.en_ko.in_house.selectstar_NNNNNN.jpg
└─── jsons
    ├── train.json
    ├── val.json
    └── test.json


### 2**. 경진대회 팀 구성 및 역할** (팀원 별로 주도적으로 참여한 부분을 중심으로 간략히 작성)

- 우승현(팀장): 저희 조는 직장인 및 대학원 팀으로서 개인 각각이 별도의 활동을 진행 하였습니다. 해당 page에서는 OCR 2조 우승현이 진행한 내용입니다.

### 3**. 경진대회 수행 절차 및 방법**

- 경진대회의 기획과 수행 및 완료 과정이 드러나게 작성해 주세요.
- 실제 경진대회를 수행한 세부적인 기간과 활동 내용, 절차 등을 포함해도 좋습니다. (e.g. 경진대회 Road Map 형식 / 경진대회 WBS 형식, 경진대회 마일스톤 형식 등)

### **4. 경진대회 수행 결과**

- 경진대회 결과물이 도출된 과정을 세부적으로 기록합니다.
- 활용된 기술(구현 방법), 핵심 기능, 검증 결과 등을 기재합니다.
- 경진대회의 과정이 잘 드러날 수 있도록 데이터 전처리 과정부터 활용까지 전체적인 프로세스를 확인할 수 있도록 단계별로 작성하시기 바랍니다. (요약적, 유기적, 논리적으로 작성할 수 있도록 유의해주세요!)
    - 탐색적 분석 및 전처리 (학습데이터 소개)
    - 모델 개요
    - 모델 선정 및 분석
    - 모델 평가 및 개선
    - 시연 결과
        - 모델 성능
        - 결과물 사진/ 시연 동영상 등 선택적 포함

### **5. 자체 평가 의견**

- 경진대회 결과물에 대한 경진대회 의도와의 부합 정도 및 실무활용 가능 정도, 계획 대비 달성도, 완성도 등 **자체적인 평가 의견**과 느낀점을 포함합니다.
- 팀 차원에서 잘한 부분과 아쉬운 점을 작성합니다. (팀 별 공통 의견 중심으로 작성하며, 분량을 고려하여 개인적인 의견은 개인 회고 부분에서 작성할 수 있도록 합니다.)
    - 잘한 점들
    - 시도 했으나 잘 되지 않았던 것들
    - 아쉬웠던 점들
    - 경진대회를 통해 배운 점 또는 시사점


## 0. Overview
### Environment
- _Write Development environment_

### Requirements
- _Write Requirements_

## 1. Competiton Info

### Overview

- _Write competition information_

### Timeline

- ex) January 10, 2024 - Start Date
- ex) February 10, 2024 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
