매우 중요한 질문입니다.  
`(1024, 1024)` 원본 이미지를 가지는 고해상도 데이터셋에서 `(256, 256)` 타일링을 적용할 때, **학습 / 평가 / 추론 / 시각화** 단계별로 어떻게 프로세스를 설계해야 하는지 단계별로 설명드립니다.

---

## ✅ 전제 조건

- 원본 이미지: `(1024, 1024)`
- 타일 크기: `256x256`, 스트라이드: `128` (중복 포함)
- 타일링 후: `(256, 256)` 타일 25개 (5x5 격자)
- 모델: 타일 단위 입력 가능 (ex. STFPM, PatchCore)
- `Preprocessor` + `Tiler` 통합 사용

---

## ✅ 1. 학습 (Training)

### 🔹 목적
- 모델이 **정상 타일**(normal tiles)만 학습하도록 함
- 타일 단위로 `loss` 계산

### 🔹 프로세스

```python
# Preprocessor with tiling
preprocessor = Preprocessor({
    "image_size": 256,
    "normalization": "imagenet",
    "tiling": {
        "tile_size": 256,
        "stride": 128
    }
})

# Dataset (transform에 tiling 포함)
train_dataset = MVTecDataset(
    root_dir="datasets/mvtec",
    category="screw",
    split="train",
    transform=preprocessor,
    mask_transform=None  # 학습 시 mask 불필요
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # 배치는 타일 기준
```

### 🔹 배치 예시
- 원본 이미지 1장 → 25개 타일
- 배치 크기 4 → 4장 이미지 → 최대 100개 타일
- `batch["image"].shape` → `(100, 3, 256, 256)`

### 🔹 주의사항
- ✅ **학습 시 타일링 O**
- ✅ **이상 이미지 제외**: `train`은 정상 이미지만 포함
- ✅ **mask 없음**: 학습에 사용 안 함

---

## ✅ 2. 평가 (Evaluation)

### 🔹 목적
- 전체 테스트 이미지에 대해 **이상 점수**(image-level), **이상 맵**(pixel-level) 계산

### 🔹 프로세스

```python
# 동일한 Preprocessor 사용 (tiling 적용)
preprocessor = Preprocessor({
    "image_size": 256,
    "normalization": "imagenet",
    "tiling": {"tile_size": 256, "stride": 128}
})

test_dataset = MVTecDataset(
    root_dir="datasets/mvtec",
    category="screw",
    split="test",
    transform=preprocessor,
    mask_transform=T.ToTensor()  # 평가 시 mask 필요
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 배치 1 (이미지 기준)
```

> 🔥 주의: `batch_size=1` → 각 배치는 **한 이미지에서 나온 모든 타일 포함**

### 🔹 모델 출력 처리

모델은 타일 단위로 추론 → **타일별 anomaly map → 원본 크기 재조합 필요**

```python
from defectvad.components.tiler import TileMerger

for batch in test_loader:
    images = batch["image"]        # (N_tiles, 3, 256, 256)
    masks = batch["mask"]          # (1, 1, 1024, 1024)
    labels = batch["label"]        # (1,)

    outputs = model(images)        # anomaly_maps: (N_tiles, 256, 256)

    # 🔧 TileMerger로 재조합
    merger = TileMerger(tile_size=256, stride=128, image_size=(1024, 1024))
    for i, (tile_map, tile_idx) in enumerate(zip(outputs["anomaly_map"], outputs["tile_indices"])):
        merger.add_tile(tile_map, tile_idx)
    full_anomaly_map = merger.merge(reduction="mean")  # 또는 "max"

    pred_score = full_anomaly_map.amax().item()
```

### 🔹 메트릭 계산
- `image_AUROC`: `pred_score` vs `label`
- `pixel_AUROC`: `full_anomaly_map` vs `mask`

---

## ✅ 3. 추론 (Inference)

### 🔹 목적
- 실시간으로 고해상도 이미지 입력 → 이상 탐지

### 🔹 프로세스

```python
# Predictor with tiling
predictor = Predictor(model)
preprocessor = Preprocessor({
    "image_size": 256,
    "normalization": "imagenet",
    "tiling": {"tile_size": 256, "stride": 128}
})

# 입력: PIL.Image (1024, 1024)
image = Image.open("high_res_defect.jpg").convert("RGB")
tiles = preprocessor(image)  # (25, 3, 256, 256)

# 추론
with torch.no_grad():
    outputs = model(tiles)  # anomaly_map: (25, 256, 256)

# 재조합
merger = TileMerger(tile_size=256, stride=128, image_size=(1024, 1024))
for map_256, idx in zip(outputs["anomaly_map"], merger.get_indices_for_shape((1024, 1024))):
    merger.add_tile(map_256, idx)
anomaly_map_1024 = merger.merge("mean")

pred_score = anomaly_map_1024.amax().item()
```

---

## ✅ 4. 시각화 (Visualization)

### 🔹 목적
- 원본 크기에서 입력, 이상 맵, 마스크 비교

### 🔹 프로세스

```python
visualizer = Visualizer(save_dir="results/vis")

visualizer.plot(
    image=image,                    # (1024, 1024, 3)
    anomaly_map=anomaly_map_1024,   # (1024, 1024)
    pred_score=pred_score,
    mask=mask_1024,                 # (1024, 1024)
    threshold=0.5
)
```

> ✅ `Visualizer`는 원본 해상도를 그대로 처리 가능

---

## ✅ 전체 프로세스 요약

| 단계 | 입력 | 전처리 | 출력 | 특이사항 |
|------|------|--------|------|----------|
| **학습** | 정상 이미지 `(1024,1024)` | 타일링 → `(256,256)` | 타일 단위 loss | 이상 이미지 제외 |
| **평가** | 테스트 이미지 + 마스크 | 타일링 → 추론 → 재조합 | `pred_score`, `anomaly_map` | `TileMerger` 필요 |
| **추론** | 실시간 고해상도 이미지 | 동일 전처리 → 재조합 | 이상 점수/맵 | 실시간 성능 고려 |
| **시각화** | 원본 크기 데이터 | 없음 | `.png` 저장 | 해상도 유지 필수 |

---

## ✅ 핵심 컴포넌트: `TileMerger`

```python
# src/defectvad/components/tile_merger.py

class TileMerger:
    def __init__(self, tile_size, stride, image_size):
        self.tile_size = tile_size
        self.stride = stride
        self.image_size = image_size
        self.h, self.w = image_size
        self.heatmap = torch.zeros(1, 1, self.h, self.w)
        self.count = torch.zeros(1, 1, self.h, self.w)

    def add_tile(self, tile: torch.Tensor, index: tuple):
        c, h, w = tile.shape
        i, j = index
        y, x = i * self.stride, j * self.stride
        y2, x2 = y + self.tile_size, x + self.tile_size
        y2 = min(y2, self.h)
        x2 = min(x2, self.w)

        self.heatmap[0, :, y:y2, x:x2] += tile.unsqueeze(0)
        self.count[0, :, y:y2, x:x2] += 1

    def merge(self, reduction="mean"):
        if reduction == "mean":
            return (self.heatmap / self.count).squeeze()
        elif reduction == "max":
            return self.heatmap.squeeze()
```

---

## ✅ 결론

| ✅ 핵심 원칙 |
|------------|
| 1. **학습/추론 전처리 동일**: `Preprocessor` 통일 |
| 2. **타일링은 전처리 단계**에서 수행 |
| 3. **평가/추론 후 재조합 필수**: `TileMerger` 사용 |
| 4. **시각화는 원본 해상도 기준** |

> ✅ 이 구조는 **PatchCore, FastFlow, DRAEM 등 고해상도 타일링 기반 모델과 완벽 호환**되며,  
> **DefectVAD의 핵심 확장성**을 보여줍니다.
