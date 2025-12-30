# DefectVAD (Defect Vision Anomaly Detection)

- 오프라인 환경 친화적이며, Anomalib과 호환되면서도 불필요한 의존성 없이 순수 PyTorch로 구현된 안정적인 VAD 학습/평가 프레임워크
- 모델은 학습 시에는 loss를 포함한 딕셔너리를, 평가/추론 시에는 pred_score, anomaly_map 등을 포함한 딕셔너리 반환
- 통일된 출력 포맷 덕분에 모델 종류에 상관없이 평가 및 추론을 공통 인터페이스로 처리 가능

## 1. 프로젝트 개요

### 1.1 목표
- SOTA Vision Anomaly Detection 평가 및 학습 프레임워크 구축
- Anomalib 라이브러리의 최신 SOTA 모델 지원 (20개 이상)
- Anomalib 라이브러리의 모델 파일들 (torch_model.py, loss.py, anomaly_map.py) 변경없이 그대로 사용
- Lightning 기반 학습/평가 엔진 구현 (lightning_model.py)을 순수 PyTorch로 전환 (BaseTrainer 구현/상속)
- 표준 벤치마크 데이터셋(MVTec, VisA, BTAD)에서 모델 성능 검증

### 1.2 제약 사항

#### 1.2.1 로컬 환경 제약 극복

로컬 실행 환경으로 보안 정책에 따른 방화벽으로 외부 네트워크 접근 제한

**문제점:**

- 사전학습 가중치 다운로드 불가: ResNet, Wide ResNet, DINOv2 등의 백본 가중치를 자동으로 다운로드할 수 없음
- 보조 데이터셋 접근 불가: DRAEM의 DTD, EfficientAD의 Imagenette2 등 필수 데이터셋 다운로드 불가
- 라이브러리 설치 제한: PyPI 접근 제한으로 의존성 관리 복잡

**해결 방안:**

- 외부 환경에서 모든 가중치 파일을 사전 다운로드하여 backbones/ 폴더에 저장
- 보조 데이터셋을 datasets/ 폴더에 사전 배치
- 최소한의 핵심 라이브러리만 사용하여 의존성 최소화

#### 1.2.2 Lightning 의존성 제거

Anomalib은 PyTorch Lightning을 학습/평가 엔진(래퍼)로 사용하지만 로컬 환경에서는 사용 제한

**문제점:**

- 라이브러리 호환성 문제: Lightning과 관련 의존성 패키지 간 버전 충돌
- 불필요한 복잡성: 단순 학습 파이프라인에 과도한 추상화
- 설치 오류: 오프라인 환경에서 Lightning 설치 실패

**해결 방안:**

- 순수 PyTorch만을 사용한 학습 파이프라인 구현
- BaseTrainer 클래스를 통한 통합 학습 인터페이스 제공
- Hook 패턴을 통한 모델별 커스터마이징 지원

```python
class BaseTrainer(ABC):

    @abstractmethod
    def training_step(self, batch):
        raise NotImplementedError

    def fit(self, train_loader, max_epochs=1, valid_loader=None):
        self.max_epochs = max_epochs
        self.max_steps = max_epochs * len(train_loader)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.configure_optimizers()
        self.configure_early_stoppers()

        self.on_fit_start()
        self.on_train_start()

        for _ in range(self.max_epochs):
            self.on_train_epoch_start()
            train_outputs = self.train(self.train_loader)
            self.on_train_epoch_end(train_outputs)

            if self.valid_loader is not None:
                self.on_validation_epoch_start()
                valid_outputs = self.validate(self.valid_loader)
                self.on_validation_epoch_end(valid_outputs)

            if self.train_early_stop or self.valid_early_stop:
                break

        self.on_train_end()
        self.on_fit_end()
```


## 2. 프로젝트 구조

### 2.1 환경 설정

`configs/paths.yaml` 예시

```yaml
path:
  backbone: xxx/backbones

  dataset: xxx/datasets
  mvtec: xxx/datasets/mvtec
  visa: xxx/datasets/visa
  btad: xxx/datasets/btad

  project: xxx/defectvad
  output: xxx/defectvad/outputs
```

```python
dataset_dir = os.environ["DATASET_DIR"]
backbone_dir = os.environ["BACKBONE_DIR"]
project_dir = os.environ["PROJECT_DIR"]
source_dir = os.environ["SOURCE_DIR"]
```

### 2.2 Dataset 폴더 구조

```
dataset/
├── mvtec/
├── visa/
├── btad/
│
├── dtd/            # draem model training
└── imagenette2/    # efficientad model training
```

### 2.3 Backbone 폴더 구조 및 파일 (`*.pth`, `model.safetensors`)

```
backbones/
├── resnet18-f37072fd.pth
├── resnet34-b627a593.pth
├── resnet50-0676ba61.pth
├── wide_resnet50_2-95faca4d.pth
├── wide_resnet50_2-32ee1156.pth
├── efficientnet_b5_lukemelas-1a07897c.pth
│
├── resnet50.tv_in1k/
│   └── model.safetensors
├── wide_resnet50_2.tv_in1k/
│   └── model.safetensors
├── deit_base_distilled_patch16_224.fb_in1k/
│   └── model.safetensors
├── deit_base_distilled_patch16_384.fb_in1k/
│   └── model.safetensors
├── cait_s24_224.fb_dist_in1k/
│   └── model.safetensors
├── cait_m48_448.fb_dist_in1k/
│   └── model.safetensors
│
├── dinov2_vits14_pretrain.pth
├── dinov2_vitb14_pretrain.pth
├── dinov2_vitl14_pretrain.pth
├── dinov2_vits14_reg4_pretrain.pth
├── dinov2_vitb14_reg4_pretrain.pth
├── dinov2_vitl14_reg4_pretrain.pth
│
└── efficientad_pretrained_weights/
    ├── pretrained_teacher_medium.pth
    └── pretrained_teacher_small.pth
```

### 2.4 Project 폴더 구조

```
defectvad/                              # project_dir
│
├── src/                                # source_dir
│   └── defectvad/
│       ├── common/                     # created files for training and evaluation
│       │   ├── __init__.py
│       │   ├── backbone.py             # get_backbone_path
│       │   ├── base_trainer.py         # BaseTrainer (pure pytorch version)
│       │   ├── config.py               # load_config, merge_configs
│       │   ├── early_stopper.py        # EarlyStopper
│       │   ├── factory.py              # create_dataset, create_dataloader, create_trainer from configs
│       │   └── seed.py                 # set_seed
│       ├── componests/                 # copied components from anomalib
│       │   ├── __init__.py
│       │   ├── feature_extractor.py    # copied from anomalib
│       │   ├── blur.py                 # copied from anomalib
│       │   ├── tiler.py                # copied from anomalib
│       │   └── *.py
│       ├── data/                       # Dataset / Dataloader / transforms
│       │   ├── __init__.py
│       │   ├── dataloaders.py          # get_train_loader, get_test_loader, get_dataloader
│       │   ├── datasets.py             # BaseDataset, MVTecDataset, ViSADataset, BTADDataset
│       │   └── trainsforms.py          # get_train_transform, get_test_transform, get_mask_transform
│       └── models/
│           ├── stfpm/
│           │   ├── __init__.py
│           │   ├── anomaly_map.py      # copied from anomalib
│           │   ├── loss.py             # copied from anomalib
│           │   ├── torch_model.py      # copied from anomalib
│           │   ├── trainer.py          # created with reference to lighting_model.py (anomalib)
│           │   └── model_config.yaml   # config for model / trainer / dataset / dataloader
│           ├── efficientad/
│           │   ├── __init__.py
│           │   ├── torch_model.py      # copied from anomalib
│           │   ├── trainer.py          # created with reference to lighting_model.py (anomalib)
│           │   └── model_config.yaml   # config for model / trainer / dataset / dataloader
│           └── model_name/
│
├── configs/                            # config_dir
│   ├── path.yaml                       # paths for backbone / dataset / project / source
│   ├── override.yaml                   # oerrides for config (ex. max_epochs, batch_size)
│   └── datasets/
│       ├── mvtec.yaml
│       ├── visa.yaml
│       └── btad.yaml
│
├── experiments/
│   ├── train_stfpm.py
│   └── train_efficientad.py
│
├── notebooks/
│   ├── eval_stfpm.ipynb
│   └── eval_efficientad.ipynb
│
├── outputs/
│
├── docs/
│
├── .gitignore
└── README.md
```


## 3. 프레임워크 사용 튜토리얼

### 3.1 모델 학습 (하드 코딩)

`experiments/train_mvtec_bottle_stfpm.py` 예시

```python
import os
import sys

def setup():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    source_dir = os.path.join(project_dir, "src")

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    os.environ["PROJECT_DIR"] = project_dir
    os.environ["SOURCE_DIR"] = source_dir
    os.environ["BACKBONE_DIR"] = "xxx/backbones"
    os.environ["DATASET_DIR"] = "xxx/datasets"


def main():
    #######################################################
    ## Initialization
    #######################################################

    from defectvad.common.seed import set_seed

    SEED = 42
    set_seed(SEED)

    #######################################################
    ## Datasets
    #######################################################

    from defectvad.data.datasets import MVTecDataset
    from defectvad.data.transforms import get_train_transform, get_test_transform, get_mask_transform

    DATASET_DIR = "xxx/datasets/mvtec"
    CATEGORY = "bottle"

    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True

    train_dataset = MVTecDataset(
        root_dir=DATASET_DIR,
        category=CATEGORY,
        split="train",
        transform=get_train_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE,),
    )
    test_dataset = MVTecDataset(
        root_dir=DATASET_DIR,
        category=CATEGORY,
        split="test",
        transform=get_test_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
        mask_transform=get_mask_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE,),
    )

    #######################################################
    ## Dataloaders
    #######################################################

    from defectvad.data.dataloaders import get_train_loader, get_test_loader

    BATCH_SIZE = 16

    train_loader = get_train_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_loader = get_test_loader(dataset=test_dataset, batch_size=BATCH_SIZE)

    #######################################################
    ## Model Trainer
    #######################################################

    from defectvad.models.stfpm.trainer import STFPMTrainer

    BACKBONE = "resnet50"
    LAYERS = ["layer1", "layer2", "layer3"]
    MAX_EPOCHS = 10

    trainer = STFPMTrainer(backbone=BACKBONE, layers=LAYERS)
    trainer.fit(train_loader, max_epochs=MAX_EPOCHS, valid_loader=test_loader)


if __name__ == "__main__":
    setup()
    main()
```

### 3.2 모델 학습 (config 사용)

**실행 예시**
```
> python experiments/train.py --dataset mvtec --category bottle --model stfpm
```

```python
# experiments/train.py

import os
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def setup():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    source_dir = os.path.join(project_dir, "src")

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    os.environ["PROJECT_DIR"] = project_dir
    os.environ["SOURCE_DIR"] = source_dir
    os.environ["BACKBONE_DIR"] = "xxx/backbones"
    os.environ["DATASET_DIR"] = "xxx/datasets"


def main(dataset_name, category_name, model_name):
    #################################################################
    # Configuration
    #################################################################

    from defectvad.common.config import load_config, merge_configs

    source_dir = os.envirion["SOURCE_DIR"]
    model_dir = os.path.join(source_dir, "defectvad", "models", model_name)
    config = load_config(os.path.join(model_dir, "model_config.yaml"))

    path_config = config["path"]
    dataset_config = config["dataset"]
    train_config = config["dataloader"]["train"]
    test_config = config["dataloader"]["test"]
    trainer_config = config["trainer"]

    dataset_config["name"] = dataset_name
    dataset_config["category"] = category_name
    dataset_config["path"] = path_config[dataset_name]
    config["name"] = model_name

    #################################################################
    # Initialization
    #################################################################

    from defectvad.common.seed import set_seed

    set_seed(config.get("seed", 42))

    #################################################################
    # Datasets / Dataloaders / Trainer
    #################################################################

    from defectvad.common.factory import create_dataset_from_config
    from defectvad.common.factory import create_dataloader_from_config
    from defectvad.common.factory import create_trainer_from_config, train_from_config

    train_dataset = create_dataset_from_config("train", dataset_config)
    test_dataset = create_dataset_from_config("test", dataset_config)

    train_loader = create_dataloader_from_config(train_dataset, train_config)
    test_loader = create_dataloader_from_config(test_dataset, test_config)

    trainer = create_trainer_from_config(trainer_config)
    train_from_config(trainer, trainer_config, train_loader, valid_loader=test_loader)


if __name__ == "__main__":
    args = parse_args()
    setup()
    main(args.dataset, args.category, args.model)
```

### 3.3 모델 평가 (예정)

```
outputs/
└── {model}/
    └── {dataset}/
        └── {category}/
            ├── weights/
            │   ├── last.pth
            │   └── best.pth
            ├── logs/
            │   └── metrics.json
            ├── predictions/
            │   └── test_results.pkl
            └── config.yaml
```
