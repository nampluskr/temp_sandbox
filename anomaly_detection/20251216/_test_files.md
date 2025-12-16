### `conftest.py`

```python
# tests/conftest.py

import os
import tempfile
import pytest
import yaml


@pytest.fixture(scope="session")
def test_config_path():
    """임시 paths.yaml 생성 및 경로 반환 (세션 단위, os.path 사용)"""
    tmpdir = tempfile.mkdtemp()

    config_dir = os.path.join(tmpdir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "paths.yaml")

    test_paths = {
        "root_dir": tmpdir,
        "source_dir": os.path.join(tmpdir, "src"),
        "dataset_dir": os.path.join(tmpdir, "datasets"),
        "backbone_dir": os.path.join(tmpdir, "backbones"),
        "output_dir": os.path.join(tmpdir, "outputs"),
    }

    with open(config_path, "w") as f:
        yaml.dump(test_paths, f)

    yield config_path

    # cleanup (tmpdir 자동 삭제는 tempfile에서 처리)
    # 필요 시 수동 정리 로직 추가 가능


@pytest.fixture(scope="session")
def test_dataset_dir():
    """임시 데이터셋 디렉터리 생성 및 더미 파일 추가 (os.path 사용)"""
    tmpdir = tempfile.mkdtemp()
    base_dataset_dir = os.path.join(tmpdir, "datasets")
    train_good_dir = os.path.join(base_dataset_dir, "mvtec", "train", "good")
    os.makedirs(train_good_dir, exist_ok=True)

    # 더미 이미지 파일 생성
    for i in range(2):
        dummy_file = os.path.join(train_good_dir, f"dummy_{i}.png")
        open(dummy_file, "w").close()  # 빈 파일 생성

    yield base_dataset_dir


@pytest.fixture(scope="session")
def test_backbone_dir():
    """임시 백본 디렉터리 및 더미 가중치 파일 생성 (os.path 사용)"""
    tmpdir = tempfile.mkdtemp()
    backbone_dir = os.path.join(tmpdir, "backbones")
    os.makedirs(backbone_dir, exist_ok=True)

    # 더미 가중치 파일 생성
    dummy_weight = os.path.join(backbone_dir, "resnet50_imagenet.pth")
    with open(dummy_weight, "wb") as f:
        f.write(b"dummy weights")

    yield backbone_dir


@pytest.fixture
def sample_config_dict():
    """테스트용 설정 딕셔너리 반환"""
    return {
        "model": "stfpm",
        "backbone": "resnet50",
        "dataset": "mvtec",
        "img_size": 256,
        "batch_size": 4,
        "learning_rate": 0.001,
    }


@pytest.fixture
def temp_output_dir():
    """임시 출력 디렉터리 제공 (각 테스트마다 독립적)"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # tempfile이 자동 정리


@pytest.fixture
def dummy_transform():
    """간단한 더미 변환 (ToTensor 위주)"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
```

## 테스트 파일 리스트

- 프로젝트 구조와 TDD 방식에 기반한 **테스트 파일 리스트**와 각 테스트 함수의 **목적 및 함수명 예시**를 정리
- 각 테스트 파일은 브랜치 개발 순서에 따라 구성되며, 함수명은 `test_` 접두어와 명확한 목적을 반영하도록 작성

1. **`test_01_config.py`**
   - `test_load_yaml_config`: YAML 파일이 정상적으로 로드되는지 확인
   - `test_resolve_relative_paths`: 상대 경로가 절대 경로로 올바르게 변환되는지 테스트
   - `test_missing_required_key_raises_error`: 필수 설정 키 누락 시 예외 발생 확인
   - `test_path_interpolation_works`: `paths.yaml` 내 변수 치환(예: `${dataset_dir}`) 정상 동작 확인

2. **`test_02_dataset.py`**
   - `test_anomaly_dataset_initialization`: AnomalyDataset 초기화 시 기본 속성 설정 확인
   - `test_dataset_returns_valid_sample`: `__getitem__`이 이미지, 라벨, 마스크를 올바른 형식으로 반환
   - `test_transforms_applied_correctly`: 전처리(transforms)가 학습/테스트에 따라 정상 적용
   - `test_train_val_test_split`: 데이터셋이 train/val/test로 올바르게 분할되는지 확인
   - `test_supported_datasets_load`: MVTec, ViSA, BTAD 데이터셋 경로 기반 자동 로드 테스트

3. **`test_03_backbone.py`**
   - `test_backbone_model_creation`: ResNet, EfficientNet 등 백본 모델 생성 가능 여부 확인
   - `test_pretrained_weights_loaded`: 사전 학습 가중치가 올바르게 로드되는지 검증
   - `test_return_layers_extraction`: `return_layers` 설정에 따라 다중 특징맵 출력 확인
   - `test_backbone_output_shapes`: 각 특징맵의 출력 크기(shape)가 예상과 일치하는지 테스트

4. **`test_04_model_base.py`**
   - `test_base_model_abstract_methods`: `train_step`, `predict`, `save`, `load` 추상 메서드 존재 확인
   - `test_model_factory_registers_classes`: 모델 팩토리에 STFPM, EfficientAD 등록 여부 테스트
   - `test_save_load_model_weights`: 모델 가중치 저장 및 로드 시 일관성 유지 확인
   - `test_predict_method_signature`: `predict` 메서드가 입력 이미지로 이상 점수 반환하는지 확인

5. **`test_05_stfpm.py`**
   - `test_stfpm_teacher_student_architecture`: 교사-학생 네트워크 구조 정의 및 동기화 테스트
   - `test_feature_pyramid_alignment`: 특징 피라미드에서 각 레이어 차이 계산 정상 동작 확인
   - `test_anomaly_map_generation`: 학생-교사 특징 차이 기반 이상 맵 생성 및 크기 검증
   - `test_stfpm_loss_calculation`: 재구성 손실(L2 등) 계산 로직 정확성 테스트

6. **`test_06_trainer.py`**
   - `test_trainer_initialization_with_model_dataloader`: Trainer가 모델과 데이터로더로 초기화 가능
   - `test_training_step_execution`: `train_step`이 한 배치에서 손실 반환 및 역전파 수행 확인
   - `test_validation_loop_runs`: 검증 루프가 주기적으로 실행되고 메트릭 기록되는지 확인
   - `test_checkpoint_saved_on_epoch`: 체크포인트가 지정된 주기로 `outputs/`에 저장되는지 테스트
   - `test_early_stopping_triggers`: 검증 성능 향상 없을 시 조기 종료 동작 확인

7. **`test_07_efficientad.py`**
   - `test_efficientad_autoencoder_structure`: 자율형 인코더(Encoder, Decoder, Subtractor) 구조 확인
   - `test_pretrained_components_loaded`: 사전 학습된 EfficientNet 가중치 로드 및 동결 테스트
   - `test_reconstruction_and_residual_map`: 재구성 이미지와 잔차 기반 이상 점수 생성 확인
   - `test_efficientad_predict_output_shape`: `predict` 출력이 입력과 동일한 해상도 유지 확인

--- 

이 테스트 계획은 **TDD 원칙**에 따라 각 기능 구현 전에 작성되어야 하며, 모든 테스트가 통과한 후에만 `main` 브랜치로 머지하는 것을 권장합니다.
