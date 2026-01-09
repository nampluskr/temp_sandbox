# docs/struture.md

### Python Interperter

```
c:/ (or d:/Non_Documents)
└── winpython/
    ├── WPy64-310111_{suffix}/          # python 3.10 / pytorch 2.2.1 / cuda 11.8
    ├── {winpython_version}_{suffix}/   # python 3.xx / pytorch 2.x.x / cuda x.x
    └── pip_list.txt                    # installed library list
```

### Data Archive

```
d:/2_SYSTEM_LEVEL
└── {measured_date}_{model_code}_{suffix}/
    ├── raw/                        # measured data (*.npz, *.png, *.log, *.mim)
    ├── measure_{option}_{measured_date}.py
    ├── measure_config.yaml         # load / update
    │
    ├── raw_rgb/                    # preprocessed rgb images
    └── infer/                      # anomoaly maps / classification results
        ├── anomaly/
        │   └─── {vad_model}/
        │        └── {filename}_{vad_model}_anomaly.png     # data / vad model info
        ├── normaly/
        │   └─── {vad_model}/
        │        └── {filename}_{vad_model}_normal.png      # data / vad model info
        │── inference_report.md
        └── inference_report.pdf    # EDM Upload / Automailing
```

### Vision Anomaly Detection (VAD) Framework

```
e:/
└── ai_inspection/                  # or 00_ai_inspection
    ├── defectvad/                  # core pytorch framework (github)
    ├── datasets/                   # reference datasets
    ├── backbones/                  # pretrained weights
    │
    ├── chamber_pipeline/           # chamber training/inference (github)
    │   ├── docs/
    │   │   └── structure.py
    │   │
    │   ├── chamber/
    │   │   ├── chamber_fake.py
    │   │   ├── sensor_fake.py    
    │   │   └── txhost_fake.py
    │   │
    │   ├── measurement/
    │   │   ├── measure.py
    │   │   └── config_sample.yaml
    │   │
    │   ├── preprocessing/
    │   │   ├── color_converter.py  # ColorConverter
    │   │   └── augment.py
    |   |
    │   ├── training/
    │   │   ├── dataset.py
    │   │   ├── dataloader.py
    │   │   └── train.py
    |   |
    │   └── prediction/
    │       ├── visualizer.py
    │       ├── predicter.py
    │       ├── reporter.py
    │       └── infer.py
    │
    └── trained_weights/            # core pytorch framework (github)
        └── {category}/             # pattern group (default category = pattern)
            └── {vad_model}/        # use recent weights
                ├── weights_{category}_{vad_model}_{trained_date}.pth
                ├── config_{category}_{vad_model}_{trained_date}.yaml
                └── log_{category}_{vad_model}_{trained_date}.txt
```

**NOTE: Python Conventions**

- 파이썬은 winpython 개별 폴더에서 독립적으로 실행되어야 함.
- 전역변수 경로 추가 금지
- 필수(최소) 라이브러리만 설치
- 경로명은 윈도우에서는 `\\` 사용하고, Linux (NAMU)에서는 `/` 사용
- 폴더명, 파일명, 변수명(명사), 함수명(동사)은 소문자만 사용하고, 단어간 연결은 밑줄 사용 (snake_case)
- 클래스명은 대문자로 시작 (CamelCase)

**NOTE: Measured Data Identification**

- `measured_date`: 측정일자 `YYYYMMDD` (8자리)
- `model_code`: 10자리
- `suffix`: 추가 정보 / 재측정 / 다른 평가

**NOTE: VAD Model Identification**

- `vad_model`: vad model name
- `category`: 초기값으로 패턴명을 사용하고, 이후 그룹핑 사용
- `timestamp`: 학습 시작 시간 (YYYYMMDD_hhmmss)
