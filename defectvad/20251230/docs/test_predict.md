# `trainer.test()` vs. `trainer.predict()`

목적과 반환값이 전혀 다름: 동작 방식, 반환 형태, 활용 예시 정리

## 1. `trainer.test(test_loader)`  

### (1) 언제 사용하나요?  
- **모델 성능을 정량적으로 평가**하고 싶을 때  
- 전체 테스트 셋에 대해 **AUROC, AUPR, F1** 등 **통계 지표**를 한 번에 구하고자 할 때  

### (2) 입력  
- `test_loader` : `torch.utils.data.DataLoader` 혹은 `LightningDataModule` 의 `test_dataloader()` 로부터 얻은 로더  
- (옵션) `ckpt_path` 를 지정하면 해당 체크포인트를 로드해서 평가할 수 있음  

### (3) 반환값  
`trainer.test()` 은 **리스트** 형태로 반환합니다. 리스트 안에는 **딕셔너리**가 하나 들어가며, 이 딕셔너리는 전체 테스트 셋에 대한 **통계 지표**만을 포함합니다.

```python
test_results = trainer.test(test_loader)   # 혹은 trainer.test(model, datamodule=dm)

# 예시 출력
print(test_results[0])
# -------------------------------------------------
# {
#   'test/auroc': 0.9873,          # 이미지 레벨 AUROC
#   'test/aupr': 0.9651,           # 이미지 레벨 AUPR
#   'test/f1': 0.921,              # 최적 threshold 에서의 F1
#   'test/image_auroc': 0.992,     # 이미지 전체에 대한 AUROC (pixel‑level aggregation)
#   'test/pixel_auroc': 0.981,     # 픽셀 레벨 AUROC
#   'test/pixel_aupr': 0.957,      # 픽셀 레벨 AUPR
#   'test/pixel_f1': 0.874        # (옵션) 픽셀 레벨 F1
# }
# -------------------------------------------------
```

- **핵심** : 이미지 자체나 anomaly map 은 반환되지 않으며, **정량적인 점수만** 제공됩니다.  
- 필요하다면 `yaml`, `json`, `csv` 로 저장해 기록할 수 있습니다.

### (4) 활용 팁
- **다중 체크포인트 비교** : 여러 체크포인트에 대해 `trainer.test()` 를 반복 실행하고, 가장 높은 AUROC/ F1 을 가진 모델을 선택합니다.  
- **학습 중 검증** : `trainer.fit()` 안에 `val_dataloader` 가 있으면 epoch 마다 자동 검증이 수행됩니다. `trainer.validate()` 로도 별도 검증이 가능합니다.  

---

## 2. `trainer.predict(test_loader)`  

### (1) 언제 사용하나요?  
- **실제 서비스·시각화** 단계에서 개별 이미지에 대한 **이상점(Anomaly Map)** 과 **이미지 전체 이상 점수**를 얻고 싶을 때  
- **배치 단위** 혹은 **단일 이미지**에 대해 결과를 바로 확인하고 시각화하고자 할 때  

### (2) 입력  
- `test_loader` : `torch.utils.data.DataLoader` (보통 `batch_size=1` 로 설정)  
- (옵션) `ckpt_path` 를 지정해 특정 체크포인트를 로드 후 추론 가능  

### (3) 반환값  
`trainer.predict()` 은 **리스트** 형태로 반환합니다. 리스트 안에 **딕셔너리**가 여러 개 들어가며, 각 딕셔너리는 **한 이미지에 대한 결과**를 담고 있습니다.

```python
predictions = trainer.predict(test_loader)

# 첫 번째 샘플 확인
sample = predictions[0]
print(sample.keys())
# dict_keys(['image_path', 'image', 'anomaly_map', 'pixel_score', 'image_score', 'label'])

# 구체적인 값
print("image_path :", sample["image_path"])
print("image_score :", sample["image_score"])          # 0~1 사이 실수, 이미지 전체 이상도
print("anomaly_map shape :", sample["anomaly_map"].shape)  # (1, H, W) 혹은 (H, W)
```

- **`image`** : 원본 이미지 Tensor (`C, H, W`)  
- **`anomaly_map`** : 픽셀‑단위 이상 점수 (0~1) – 시각화에 바로 사용 가능  
- **`pixel_score`** : `anomaly_map` 과 동일한 값 (alias)  
- **`image_score`** : 이미지 전체 이상도 (보통 `max(anomaly_map)` 혹은 `mean(anomaly_map)` 로 계산)  
- **`label`** : 테스트셋에 라벨이 있으면 제공 (`0` 정상, `1` 결함)  

### (4) 활용 팁
- **시각화**  
  ```python
  import matplotlib.pyplot as plt

  img = sample["image"].permute(1, 2, 0).cpu().numpy()
  amap = sample["anomaly_map"].squeeze().cpu().numpy()

  fig, ax = plt.subplots(1, 2, figsize=(8, 4))
  ax[0].imshow(img)
  ax[0].set_title("Original")
  ax[0].axis("off")

  ax[1].imshow(amap, cmap="jet")
  ax[1].set_title(f"Anomaly Map (score={sample['image_score']:.3f})")
  ax[1].axis("off")
  plt.show()
  ```
- **배치 추론** : `batch_size` 를 크게 잡아도 `predict` 가 반환하는 리스트는 **배치 순서대로** 결과를 담고 있으니, `for` 루프를 돌며 저장하거나 후처리하면 됩니다.  
- **서비스 연동** : `predict` 결과를 바로 API 응답으로 반환하거나, `anomaly_map` 을 PNG/JPEG 로 저장해 프론트엔드에 전달할 수 있습니다.  

---

## 3. `test` 와 `predict` 의 차이 한눈에 보기

| 항목 | `trainer.test()` | `trainer.predict()` |
|------|-------------------|----------------------|
| **목적** | 전체 테스트 셋에 대한 **정량적 성능 평가** | 개별 이미지에 대한 **이상점 지도 & 점수** 반환 |
| **입력** | `test_loader` (또는 `datamodule`) | `test_loader` (보통 `batch_size=1`) |
| **반환값** | `list[dict]` → **통계 지표만** (AUROC, AUPR, F1 등) | `list[dict]` → **이미지, anomaly_map, image_score, label** 등 |
| **시각화** | 직접 구현 필요 (지표만) | `anomaly_map` 을 바로 `matplotlib` 등으로 시각화 가능 |
| **콜백 연동** | `ImageLogger` 등은 동작하지 않음 | `ImageLogger` 는 `fit` 단계에서만 동작, `predict` 결과는 직접 시각화 |
| **주 사용 시점** | 모델 개발·튜닝 후 **평가** 단계 | **데모, 서비스, 결과 확인** 단계 |

---

## 4. 부가 기능 – 시각화 콜백 (옵션)

`trainer.fit()` 중에 `ImageLogger` 혹은 `AnomalyMapLogger` 콜백을 추가하면 **epoch 마다** 자동으로 이미지와 anomaly map 을 저장할 수 있습니다.

```yaml
# configs/callbacks.yaml
callbacks:
  - class_path: anomalib.utils.callbacks.ImageLogger
    init_args:
      log_dir: outputs/visuals
      frequency: 1          # 매 epoch마다 저장
      max_images: 4
```

`Trainer` 를 만들 때 `callbacks = get_callbacks(cfg)` 로 불러오면 자동 적용됩니다.  
하지만 `trainer.test()` 나 `trainer.predict()` 에서는 이 콜백이 동작하지 않으니, **예측 결과를 직접 시각화**하거나 **`plot_anomaly_map`** 유틸리티를 사용해야 합니다.

```python
from anomalib.utils.visualization import plot_anomaly_map
plot_anomaly_map(sample["anomaly_map"], save_path="outputs/visuals/bottle_001.png")
```

---

## 5. 정리

- **`trainer.test(test_loader)`** → 전체 테스트 셋에 대한 **AUROC / AUPR / F1** 등 **통계값**을 반환한다.  
- **`trainer.predict(test_loader)`** → 각 이미지마다 **원본, anomaly map, 이미지 점수** 등을 반환한다.  
- 정량적 평가는 `test`, 시각·서비스용 결과는 `predict` 를 사용한다.  
- 필요에 따라 `ImageLogger` 같은 콜백을 `fit` 단계에 넣어 자동 시각화를 할 수 있고, `predict` 결과는 직접 `matplotlib` 혹은 `anomalib.utils.visualization` 로 시각화한다.

