# Defect Vision Anomaly Detection

## Setup

### 1. Config 파일 수정

**`configs/defaults.yaml` 경로 수정**

```yaml
path:
  backbone: /home/namu/myspace/NAMU/backbones
  dataset: /home/namu/myspace/NAMU/datasets
  mvtec: /home/namu/myspace/NAMU/datasets/mvtec
  visa: /home/namu/myspace/NAMU/datasets/visa
  btad: /home/namu/myspace/NAMU/datasets/btad
```

**운영체제별 Dataloader 옵션 변경**

```yaml
train_loader:
  batch_size: 16
  shuffle: true
  drop_last: true
  num_workers: 0      # Windows 0 / Linux 8
  pin_memory: false   # Windows false / Linux true

test_loader:
  batch_size: 1
  shuffle: false
  drop_last: false
  num_workers: 0      # Windows 0 / Linux 8
  pin_memory: false   # Windows false / Linux true
```

### 2. `datasets` 폴더 구조

```
dataset/
├── mvtec/
├── visa/
├── btad/
│
├── dtd/            # draem model training
└── imagenette2/    # efficientad model training
```

### 3. `bacbones` 폴더 구조

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

### 4. `outputs` 폴더 구조

```
outputs/
└── {dataset}/
    └── {category}/
        └── {model}/
            ├── anomaly/
            ├── normal/
            ├── weights_{dataset}_{category}_{model}_{max_epochs}epoch.pth
            └── configs_{dataset}_{category}_{model}_{max_epochs}epoch.yaml
```

## Project 폴더 구조

```
defectvad/                              # project_dir
│
├── src/                                # source_dir
│   └── defectvad/
│       ├── common/                     # created files for training and evaluation
│       │   ├── __init__.py
│       │   ├── backbone.py             # get_backbone_path
│       │   ├── base_model.py           # BaseModel (pytorch version)
│       │   ├── base_trainer.py         # BaseTrainer (pytorch version)
│       │   ├── config.py               # load_config, merge_configs
│       │   ├── early_stopper.py        # EarlyStopper (pytorch version)
│       │   ├── evaluator.py            # Evaluator (pytorch version)
│       │   ├── factory.py              # create dataset / dataloader / model / trainer
│       │   ├── visualizer.py           # Visualier (pytorch version)
│       │   └── utils.py                # set_seed
│       ├── componests/                 # copied components from anomalib
│       │   ├── __init__.py
│       │   ├── feature_extractor.py    # copied from anomalib
│       │   ├── blur.py                 # copied from anomalib
│       │   ├── tiler.py                # copied from anomalib
│       │   └── *.py                    # copied from anomalib
│       ├── data/                       # Dataset / transforms
│       │   ├── __init__.py
│       │   ├── datasets.py             # BaseDataset, MVTecDataset, ViSADataset, BTADDataset
│       │   └── trainsforms.py          # get_image_transform, get_mask_transform
│       └── models/
│           ├── stfpm/
│           │   ├── __init__.py
│           │   ├── anomaly_map.py      # copied from anomalib
│           │   ├── loss.py             # copied from anomalib
│           │   ├── torch_model.py      # copied from anomalib
│           │   └── model_trainer.py    # created with reference to lighting_model.py (anomalib)
│           ├── efficientad/
│           │   ├── __init__.py
│           │   ├── torch_model.py      # copied from anomalib
│           │   └── model_trainer.py    # created with reference to lighting_model.py (anomalib)
│           └── {model}/                # models from anomalib
│
├── configs/                            # config_dir
│   ├── defaults.yaml                   # seed, paths for backbones / datasets
│   ├── datasets/                       # configs for datasets / transforms / dataloaders
│   │   ├── mvtec.yaml
│   │   ├── visa.yaml
│   │   └── btad.yaml
│   └── models/                         # configs for model and trainer
│       ├── efficientad.yaml
│       ├── stfpm.yaml
│       └── {model}.yaml
│
├── experiments/
│   ├── run.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
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
