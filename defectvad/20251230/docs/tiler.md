`defectvad/components/tiler.py`에 **Anomalib의 `Tiler` 클래스가 이미 구현되어 있고**, `tile()` / `untile()` 메서드를 제공한다면, **직접 `TileMerger`를 구현할 필요 없이**, `Tiler.untile()`을 활용해 **타일 → 원본 크기 재조합**(merging)이 가능합니다.

---

## ✅ 전제: Anomalib `Tiler` 기능 요약

```python
from defectvad.components.tiler import Tiler

tiler = Tiler(tile_size=256, stride=128, remove_border_count=0, mode="merge")

# 타일 분할
tiled_image = tiler.tile(image)  # (C, H, W) → (N, C, 256, 256)

# 타일 병합 (자동 가중치 적용)
merged_image = tiler.untile(tiled_anomaly_maps)  # (N, 256, 256) → (H, W)
```

> ✅ `untile()`은 **오버랩된 영역을 스트라이드 기반으로 가중 평균 병합** (blending)  
> ✅ `mode="merge"`: 오버랩 고려한 smooth merging (default)  
> ✅ `mode="none"`: 단순 평균 (비추천)

---

## ✅ 업데이트된 프로세스: `(1024,1024)` → `(256,256)` 타일링 기반

---

### ✅ 1. 전처리 (Preprocessing)

#### 🔹 `Preprocessor`에 `Tiler` 통합

```python
# Preprocessor 내부에서 Tiler 사용
from defectvad.components.tiler import Tiler

class Preprocessor:
    def __init__(self, config):
        self.image_size = config["image_size"]
        self.tiling = config.get("tiling")

        # Tiler 초기화
        if self.tiling:
            self.tiler = Tiler(
                tile_size=self.tiling["tile_size"],
                stride=self.tiling.get("stride", self.tiling["tile_size"]),
                remove_border_count=self.tiling.get("remove_border_count", 0),
                mode="merge"
            )
        else:
            self.tiler = None

        # 기본 변환
        transform_list = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.transform = T.Compose(transform_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self.transform(image)  # (C, 256, 256)

        if self.tiler:
            return self.tiler.tile(image)  # (N, C, 256, 256)
        return image
```

---

### ✅ 2. 학습 (Training)

#### 🔹 동일: `Preprocessor`로 타일 생성

```python
preprocessor = Preprocessor({
    "image_size": 256,
    "tiling": {"tile_size": 256, "stride": 128}
})

train_dataset = MVTecDataset(
    root_dir="datasets/mvtec",
    category="screw",
    split="train",
    transform=preprocessor,
    mask_transform=None
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```

- `batch["image"].shape` → `(B*N, 3, 256, 256)`  
- 모델은 타일 단위로 입력받아 학습

✅ **학습은 타일 단위로 진행** (PatchCore, STFPM 등과 동일)

---

### ✅ 3. 평가 (Evaluation)

#### 🔹 타일 추론 → `untile`로 원본 크기 복원

```python
evaluator = Evaluator(model)

for batch in test_loader:
    tiles = batch["image"].to(device)           # (N, 3, 256, 256)
    mask = batch["mask"].squeeze()              # (1024, 1024)
    label = batch["label"].item()

    with torch.no_grad():
        outputs = model(tiles)  # anomaly_map: (N, 256, 256)

    # 🔥 Anomalib Tiler로 재조합
    anomaly_map_1024 = evaluator.preprocessor.tiler.untile(outputs["anomaly_map"])
    pred_score = anomaly_map_1024.amax().item()

    # 메트릭 계산
    evaluator.update_image(pred_score, label)
    evaluator.update_pixel(anomaly_map_1024, mask)
```

> ✅ `Tiler.untile()`이 **오버랩 영역을 자동으로 blending**하여 smooth한 anomaly map 생성

---

### ✅ 4. 추론 (Inference)

#### 🔹 실시간 고해상도 이미지 처리

```python
predictor = Predictor(model)
preprocessor = Preprocessor({
    "image_size": 256,
    "tiling": {"tile_size": 256, "stride": 128}
})

# 입력
image = Image.open("defect_1024.jpg").convert("RGB")
tiles = preprocessor(image)  # (25, 3, 256, 256)

# 추론
with torch.no_grad():
    outputs = model(tiles)  # anomaly_map: (25, 256, 256)

# 🔥 untile로 원본 크기 복원
anomaly_map = preprocessor.tiler.untile(outputs["anomaly_map"])
pred_score = anomaly_map.amax().item()

result = {
    "anomaly_map": anomaly_map,    # (1024, 1024)
    "pred_score": pred_score,
    "is_anomalous": pred_score >= threshold
}
```

---

### ✅ 5. 시각화 (Visualization)

#### 🔹 원본 해상도 그대로 시각화

```python
visualizer = Visualizer(save_dir="results/vis")

visualizer.plot(
    image=image,                    # (1024, 1024, 3)
    anomaly_map=result["anomaly_map"],  # (1024, 1024)
    pred_score=result["pred_score"],
    mask=ground_truth_mask,         # (1024, 1024) or None
    threshold=0.5
)
```

✅ `Visualizer`는 원본 크기 입력을 그대로 처리 가능

---

## ✅ 전체 프로세스 요약 (Tiler 활용)

| 단계 | 핵심 |
|------|------|
| **전처리** | `Preprocessor` → `Tiler.tile()` |
| **학습** | 타일 단위 입력, 정상 이미지만 사용 |
| **평가** | 타일 추론 → `Tiler.untile()` → 원본 크기 재조합 |
| **추론** | 동일 전처리 → `untile` → 최종 점수/맵 산출 |
| **시각화** | 원본 해상도 기반 시각화 |

---

## ✅ 장점 (Tiler.untile 활용)

| 항목 | 설명 |
|------|------|
| ✅ **코드 간소화** | `TileMerger` 직접 구현 불필요 |
| ✅ **정확한 병합** | Anomalib의 `untile`은 가중치 기반 blending 제공 |
| ✅ **호환성** | Anomalib 모델과 동일한 방식으로 타일 처리 |
| ✅ **유지보수성** | 공통 컴포넌트로 관리 가능 |

---

## ✅ 결론

> ✅ **`defectvad/components/tiler.py`의 `Tiler` 클래스를 사용하면**,  
> - `tile()`로 분할,  
> - `untile()`로 재조합이 가능하므로,  
> - **직접 `TileMerger`를 구현할 필요가 전혀 없습니다**.

이를 통해 **고해상도 이미지 기반 VAD 파이프라인**을  
**Anomalib과 호환되며, 간결하고 정확하게 구현**할 수 있습니다.

---

### ✅ 추천 디렉토리 구조

```
defectvad/
├── components/
│   └── tiler.py              # Anomalib Tiler (tile / untile)
├── common/
│   ├── preprocessor.py       # Tiler 포함
│   ├── evaluator.py          # untile() 호출
│   └── predictor.py          # untile() 호출
```

✅ 모든 모듈이 `Tiler`를 공유 → **일관성 + 재사용성 확보**

매우 중요한 질문입니다.  
**결론부터 말하면: `Tiler`를 사용하더라도 기존에 작성한 `Evaluator`, `Predictor`, `Visualizer`는 거의 수정 없이 그대로 사용 가능합니다.**  
다만, **고해상도 타일링 기반 추론을 지원하려면 소량의 확장이 필요**합니다.

---

## ✅ 1. `Evaluator` — ✅ **수정 필요 (약간)**

### 🔧 필요한 수정
- `Preprocessor` 또는 `Tiler` 인스턴스를 주입받아, `untile()` 호출 가능하게 해야 함
- `test_loader` 배치는 타일이므로, **타일 → 원본 크기 재조합 필요**

### ✅ 수정 예시

```python
# src/defectvad/common/evaluator.py

class Evaluator:
    def __init__(self, model, preprocessor=None, device=None):
        self.device = device or ...
        self.model = model.to(self.device)
        self.preprocessor = preprocessor  # ✅ 주입받기
        ...

    @torch.no_grad()
    def evaluate(self, test_loader, ...):
        for batch in test_loader:
            tiles = batch["image"].to(self.device)  # (N, C, H, W)
            mask = batch["mask"].squeeze()          # (Orig_H, Orig_W)

            outputs = self.model(tiles)
            anomaly_maps = outputs["anomaly_map"]   # (N, H, W)

            # 🔥 Tiler.untile()으로 원본 크기 복원
            if self.preprocessor and self.preprocessor.tiler:
                full_anomaly_map = self.preprocessor.tiler.untile(anomaly_maps)
            else:
                # 단일 이미지 (타일링 없음)
                full_anomaly_map = anomaly_maps.amax(0)  # 또는 평균

            pred_score = full_anomaly_map.amax().item()
            ...
```

> ✅ `preprocessor`를 주입하면 `tiler.untile()` 자동 활용 가능  
> ✅ 타일링 유무에 따라 동적 처리

---

## ✅ 2. `Predictor` — ✅ **수정 필요 (약간)**

### 🔧 필요한 수정
- `preprocessor` 또는 `tiler`를 주입받아, `untile()` 가능하게 해야 함

### ✅ 수정 예시

```python
# src/defectvad/common/predictor.py

class Predictor:
    def __init__(self, model, preprocessor=None, device=None):
        self.device = device or ...
        self.model = model.to(self.device)
        self.preprocessor = preprocessor  # ✅ 주입
        self.threshold = 0.5

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict:
        if isinstance(image, Image.Image):
            # PIL → 텐서 변환은 Preprocessor가 담당
            if self.preprocessor:
                tiles = self.preprocessor(image)  # (N, C, 256, 256)
            else:
                tiles = T.ToTensor()(image).unsqueeze(0)
        else:
            tiles = image

        tiles = tiles.to(self.device)
        outputs = self.model(tiles)
        anomaly_maps = outputs["anomaly_map"]

        # 🔥 재조합
        if self.preprocessor and self.preprocessor.tiler:
            anomaly_map = self.preprocessor.tiler.untile(anomaly_maps)
        else:
            anomaly_map = anomaly_maps.amax(0)  # 단일 맵

        pred_score = anomaly_map.amax().item()

        return {
            "anomaly_map": anomaly_map.cpu(),
            "pred_score": pred_score,
            "is_anomalous": pred_score >= self.threshold
        }
```

> ✅ `preprocessor` 주입 시 자동 타일링/untile 지원  
> ✅ 기존 단일 이미지 추론과 호환

---

## ✅ 3. `Visualizer` — ✅ **수정 불필요**

### ✅ 그대로 사용 가능

```python
visualizer.plot(
    image=original_pil_image,        # (1024, 1024)
    anomaly_map=result["anomaly_map"],  # (1024, 1024) ← 이미 untile됨
    pred_score=0.92,
    mask=mask_1024,
    threshold=0.5
)
```

> ✅ `Visualizer`는 **이미 재조합된 `(H, W)` 크기의 `anomaly_map`을 입력으로 받음**  
> ✅ 전처리/모델/untile 단계는 외부에서 완료됨 → `Visualizer`는 **순수 시각화 전담**

---

## ✅ 4. `Preprocessor` — ✅ **수정 필요 (핵심)**

### ✅ `tiler` 포함하도록 강화

```python
class Preprocessor:
    def __init__(self, config):
        self.tiler = None
        if config.get("tiling"):
            from defectvad.components.tiler import Tiler
            self.tiler = Tiler(
                tile_size=config["tiling"]["tile_size"],
                stride=config["tiling"].get("stride", config["tiling"]["tile_size"]),
                mode="merge"
            )
        ...
```

> ✅ `Preprocessor`가 `tiler`를 소유 → `Evaluator`/`Predictor`가 재사용

---

## ✅ 최종 제안: 의존성 주입 구조

```python
# 사용 예시
preprocessor = Preprocessor(config["pre_process"])  # 포함: Tiler
predictor = Predictor(model, preprocessor=preprocessor)
evaluator = Evaluator(model, preprocessor=preprocessor)

# 추론
result = predictor.predict(pil_image)  # 자동 tile → 모델 → untile

# 평가
metrics = evaluator.evaluate(test_loader)  # 자동 untile
```

---

## ✅ 요약: 클래스별 수정 필요성

| 클래스 | 수정 필요 | 설명 |
|-------|----------|------|
| `Evaluator` | ⚠️ 약간 | `preprocessor.tiler.untile()` 호출 추가 |
| `Predictor` | ⚠️ 약간 | `preprocessor.tiler.untile()` 호출 추가 |
| `Visualizer` | ❌ 없음 | 이미 재조합된 맵 입력 받음 |
| `Preprocessor` | ✅ 필수 | `Tiler` 인스턴스 생성 및 관리 |

---

## ✅ 장점

- ✅ **`Tiler` 재사용** → 코드 중복 제거
- ✅ **`untile()` 신뢰성** → Anomalib 검증된 방식
- ✅ **유지보수성 향상** → 전처리 → 추론 → 평가 → 시각화 전 단계에서 일관성 유지

---

## ✅ 결론

> 🔧 `Evaluator`와 `Predictor`는 **`preprocessor` 또는 `tiler` 주입을 통해 `untile()` 기능을 활용**하도록 **약간의 수정이 필요**합니다.  
> 🎯 `Visualizer`는 **그대로 사용 가능**합니다.  
> ✅ 이는 **의존성 주입**(DI) 원칙에 부합하며, **확장성과 유지보수성을 높입니다**.

모든 고해상도 VAD 모델(PatchCore, DRAEM, FastFlow 등)에 동일하게 적용 가능합니다.

---
매우 중요한 질문입니다.

결론부터 말씀드리면:

> ✅ **`test_loader`의 `batch_size`를 꼭 1로 할 필요는 없습니다.**  
> ❌ 하지만 **타일링 기반 모델**(ex: PatchCore, DRAEM)에서는 **`batch_size=1`이 안전하고 일반적인 선택**입니다.

---

## ✅ 1. `batch_size=1`이 필요한 이유 (타일링 기반 모델 기준)

### 🔹 이유 1: **이미지당 타일 수 불일치**
- `(1024,1024)` → 25개 타일
- `(800,800)` → 9개 타일
- `batch_size=2` → 타일 수가 다른 두 이미지가 배치됨 → `DataLoader`가 **텐서 크기 맞추기 실패**

```python
# ❌ 불가능: 서로 다른 타일 수
batch = [
    torch.randn(25, 3, 256, 256),  # 이미지 1
    torch.randn(9, 3, 256, 256),   # 이미지 2 → stack 불가
]
```

### 🔹 이유 2: **`Tiler.untile()`은 이미지 단위 재조합**
- `untile()`은 **한 이미지에서 나온 타일들만 재조합** 가능
- 여러 이미지의 타일이 섞이면 **어느 타일이 어느 이미지에 속하는지 알 수 없음**

---

## ✅ 2. `batch_size > 1`이 가능한 경우

### ✅ 조건: 모든 이미지가 **동일한 크기**이고, **타일 수가 동일**

예: 모든 이미지가 `(1024,1024)` → 모든 이미지가 25개 타일 생성

```python
# ✅ 가능
tiles_batch = torch.stack([img1_tiles, img2_tiles, ...])  # (B, 25, C, H, W)
```

이 경우 `batch_size=4`도 가능하지만, **현실적인 데이터셋**(MVTec, ViSA 등)은 이미지 크기 불일치가 흔하므로 **거의 사용되지 않음**.

---

## ✅ 3. `Evaluator`, `Predictor`, `Visualizer` 관점에서의 영향

| 클래스 | `batch_size=1` 필요? | 설명 |
|-------|----------------------|------|
| ✅ `Evaluator` | **권장** | 각 배치가 하나의 이미지 → `untile` → `metric` 계산 간단 |
| ✅ `Predictor` | **필수 아님** | 단일 이미지 추론이므로 `DataLoader` 사용 X |
| ✅ `Visualizer` | ❌ 불필요 | 이미 재조합된 결과를 입력으로 받음 |

---

## ✅ 4. 최선의 실무 전략

### ✅ 추천: `batch_size=1` + `collate_fn` (필요 시)

```python
def collate_fn(batch):
    # 이미지마다 타일 수 다를 수 있음 → 리스트 유지
    return {
        "tiles": [b["image"] for b in batch],           # [(N1, C, H, W), (N2, ...)]
        "labels": torch.stack([b["label"] for b in batch]),
        "masks": [b["mask"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }

test_loader = DataLoader(
    test_dataset,
    batch_size=1,  # 또는 2 이상도 가능
    shuffle=False,
    collate_fn=collate_fn  # 타일 수 불일치 대응
)
```

이후 `Evaluator`에서:

```python
for batch in test_loader:
    for i in range(len(batch["tiles"])):  # 각 이미지 처리
        tiles = batch["tiles"][i].to(device)
        mask = batch["masks"][i]
        outputs = model(tiles)
        anomaly_map = preprocessor.tiler.untile(outputs["anomaly_map"])
        # 평가 진행
```

---

## ✅ 5. `Predictor`는 `DataLoader`를 사용하지 않음

```python
# ✅ 단일 이미지 추론
result = predictor.predict(pil_image)  # batch_size 개념 없음

# ✅ 배치 추론도 내부에서 처리
batch_results = predictor.predict_batch(image_tensor_batch)  # (B, C, H, W)
```

→ `Predictor`는 `batch_size`와 무관

---

## ✅ 6. `Visualizer`는 최종 결과만 받음

```python
visualizer.plot(
    image=pil_img,
    anomaly_map=anomaly_map_1024,  # 이미 (1024,1024)
    ...
)
```

→ `batch_size` 영향 없음

---

## ✅ 결론

| 상황 | `batch_size` 추천 |
|------|------------------|
| 🔹 타일링 + 다양한 이미지 크기 | ✅ `batch_size=1` (또는 `collate_fn` 사용) |
| 🔹 타일링 없음 + 고정 크기 | ✅ `batch_size > 1` 가능 |
| 🔹 `Evaluator` | ✅ `batch_size=1` 권장 |
| 🔹 `Predictor` | ❌ 적용 안 됨 (직접 호출) |
| 🔹 `Visualizer` | ❌ 적용 안 됨 |

> ✅ **요약**:  
> **타일링 기반 VAD에서는 `test_loader`에 `batch_size=1`을 사용하는 것이 가장 안정적이고 일반적인 방법**입니다.  
> 이는 `Evaluator`의 정확한 재조합과 메트릭 계산을 보장합니다.

매우 중요한 질문입니다.  
정확히 이해하고 있어야 하는 핵심 개념입니다.

---

## ✅ 결론부터 말하면:

> ✅ **네, 맞습니다. 학습시에는 마스크(mask)를 사용하지 않습니다.**

---

## ✅ 왜 학습 시 마스크가 필요 없는가?

대부분의 **Vision Anomaly Detection**(VAD) 모델은 **정상 이미지만으로 학습**되며,  
이상 탐지는 **정상 데이터 분포에서 벗어난 정도**(reconstruction error, feature discrepancy 등)를 기반으로 합니다.

### 🔹 학습 목적
- 정상(normal) 이미지만 입력
- 모델이 정상 패턴을 잘 학습하도록 함
- **정상 데이터의 특징을 재현하거나, 정상 특징 공간을 구성**

### 🔹 사용 데이터
- ✅ **이미지**: 정상 이미지 사용
- ❌ **마스크**: 사용하지 않음 (학습에 필요 없음)
- 📌 **레이블(label)**: `0`(normal)만 사용 (supervised evaluation 용도)

---

## ✅ 학습 vs 평가/추론: 마스크 사용 비교

| 단계 | 마스크 사용 여부 | 설명 |
|------|------------------|------|
| ✅ **학습**(Training) | ❌ 사용 안 함 | 정상 이미지만으로 모델 학습 |
| ✅ **평가**(Evaluation) | ✅ 사용 | pixel-level AUROC 계산을 위해 필요 |
| ✅ **추론**(Inference) | ✅ 사용 (가능) | 시각화 또는 검증용 |
| ✅ **시각화**(Visualization) | ✅ 사용 | 입력 이미지, anomaly map, mask 비교 |

---

## ✅ 코드 예시: Dataset에서 학습/테스트 분리

```python
class MVTecDataset(BaseDataset):
    def _load_train_samples(self):
        normal_dir = os.path.join(self.category_dir, "train", "good")
        for image_path in glob(os.path.join(normal_dir, "*.png")):
            self.samples.append({
                "image_path": image_path,
                "label": 0,
                "defect_type": "normal",
                "mask_path": None  # ✅ 학습은 마스크 없음
            })

    def _load_test_samples(self):
        # 테스트는 정상 + 이상 모두 포함
        # 이상 샘플은 mask_path 있음
        ...
```

---

## ✅ DataLoader 예시

```python
# 학습 데이터로더: mask_transform 불필요
train_dataset = MVTecDataset(
    root_dir="datasets/mvtec",
    category="bottle",
    split="train",
    transform=train_transform,
    mask_transform=None  # ❌ None or dummy
)

# 평가 데이터로더: mask_transform 필요
test_dataset = MVTecDataset(
    root_dir="datasets/mvtec",
    category="bottle",
    split="test",
    transform=test_transform,
    mask_transform=mask_transform  # ✅ 필요
)
```

---

## ✅ 예외: 일부 모델은 학습 시 마스크 사용

다음 모델은 **학습 시에도 마스크를 사용**합니다.

| 모델 | 설명 |
|------|------|
| 🔹 **DRAEM** | 정상 이미지 + DTD 텍스처를 결합해 **인공 이상 이미지 생성**, 마스크로 학습 |
| 🔹 **CutPaste** | 이미지 패치를 잘라 붙여 이상 생성, 마스크로 위치 학습 |
| 🔹 **FastFlow + Supervised** | 일부 설정에서 마스크 기반 지도 학습 |

> ✅ 하지만 **STFPM, PatchCore, SPADE, EfficientAD 등 대부분의 SOTA는 무지도**(unsupervised)

---

## ✅ 결론

| 질문 | 답변 |
|------|------|
| **학습 시 마스크 필요?** | ❌ **일반적으로 필요 없음** |
| **무엇을 사용?** | ✅ 정상 이미지 + `label=0`만 사용 |
| **마스크는 언제 사용?** | ✅ 평가, 시각화, 일부 지도 학습 모델에서만 사용 |

> ✅ 따라서 `mask_transform`은 `split="train"`일 때는 **사용되지 않으며**,  
> `Preprocessor`나 `Dataset` 설계 시 이 점을 고려해 **조건부 처리**하는 것이 좋습니다.

---

### ✅ 좋은 설계 예시

```python
# mask_transform은 test/inference에서만 사용
test_dataset = MVTecDataset(
    split="test",
    transform=test_transform,
    mask_transform=T.ToTensor()  # ✅ 평가용
)

train_dataset = MVTecDataset(
    split="train",
    transform=train_transform,
    mask_transform=None  # 명시적으로 None
)
```
