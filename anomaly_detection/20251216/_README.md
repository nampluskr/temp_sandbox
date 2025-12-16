# Anomaly Detection

- 목표: SOTA Vision Anomaly Detection 평가 및 학습 프레임워크를 pytest 기반 TDD 방식으로 개발

## Project Stucture
```
projects/
├── datasets/                       # 이상 탐지 데이터셋 저장 디렉터리
│   ├── mvtec/                      # MVtec (train/test/ground_truth)
│   ├── visa/                       # ViSA (images/annotations)
│   └── btad/                       # BTAD
├── backbones/                      # 사전 학습 백본 가중치 저장
│   └── *.pth
└── anomaly_detection/              # 프로젝트 메인 디렉터리
    ├── .gitignore                  # git 제외 대상
    ├── pytest.ini                  # pytest 설정 (테스트 경로, 커버리지 등)
    ├── README.md                   # 프로젝트 개요 및 사용 안내
    ├── src/anmomaly_detection/     # 핵심 소스 코드 (Level-1) 이후 확장 예정
    │   ├── __init__.py             # 패키지 등록
    │   ├── config.py               # 
    │   ├── datasets.py             # 데이터셋 / 데이터로더 구현
    │   ├── trainer.py              # 학습/검증 (모델 래퍼)
    │   └── models/                 # 모델 구현
    │       ├── init.py             # 모델 모듈 패키지 등록
    │       ├── stfpm.py
    │       └── effieienctad.py
    ├── tests/
    │   ├── init.py                 # 테스트 패키지 표시
    │   ├── conftest.py             # 공통 fixture 정의
    │   ├── test_01_dataset.py
    │   └── test_*.py
    ├── experiments/                # 모델별 학습/평가 스크립트
    │   ├── init.py                 # 패키지 인식용
    │   ├── train_*.py              # 모델 학습 스크립트
    │   └── eval_*.py               # 모델 평가 스크립트
    ├── notebooks/                  # 분석 및 데모용 Jupyter 노트북
    ├── outputs/                    # 생성물 저장 (체크포인트, 결과, 로그 등)
    ├── docs/                       # 문서 (설계, API, 사용법)
    ├── configs/                    # 설정 관리
    │   ├── paths.yaml              # 경로 정의
    │   ├── stfpm.yaml              # STFPM 실험 설정
    │   ├── efficientad.yaml        # EfficientAD 실험 설정
    │   └── *.yaml
    └── .git/
```

- **패키지 설정 폴더 리스트**
```
anomaly_detection/src/anomaly_detection/__init__.py
anomaly_detection/src/anomaly_detection/models/__init__.py
anomaly_detection/tests/__init__.py
anomaly_detection/experiments/__init__.py
```    

- **테스트 파일 리스트**
```
tests/test_01_config.py
tests/test_02_dataset.py
tests/test_03_backbone.py
tests/test_04_model_base.py
tests/test_05_stfpm.py
tests/test_06_trainer.py
tests/test_07_efficientad.py
```    

## TODO List (브랜치 단위)

- 각 브랜치는 독립적으로 개발 및 테스트 완료 후 `main` 브랜치로 머지됨
- TDD 원칙에 따라 **모든 구현 전에 테스트 작성**이 선행되어 함.


0. **초기 프로젝트 생성 브랜치 (`feature/init-project`)**
    - projects/anomaly_detection 디렉터리 생성
    - 프로젝트 폴더 내 git init 실행
    - .gitignore 작성 (outputs/, pycache, .pyc, .ipynb, .env 등 포함)
    - 모든 하위 디렉터리 생성 (src/, tests/, configs/, experiments/, notebooks/, docs/, outputs/)
    - 각 패키지 디렉터리에 __init__.py 파일 생성
    - 테스트 파일 생성 (내용은 빈 상태 또는 pass 포함)

1. **공통 기반 설정 브랜치 (`feature/setup`)**
    - pytest.ini 파일 작성 및 pytest 명령어 실행 검증
    - configs/paths.yaml 파일 작성 (root_dir, source_dir, dataset_dir, backbone_dir, output_dir 등 정의)
    - 초기 설정 파일 참조 테스트 추가 (예: 경로 문자열 치환 확인)

2. **설정 관리 기능 브랜치 (`feature/config-loader`)**
   - `tests/test_config.py` 작성: YAML 로드 및 변수 치환 테스트
   - `src/anomaly_detection/config.py` 구현: 설정 로드 및 경로 절대화 (os.path 사용)
   - 설정 값 유효성 검증 테스트 추가 (예: 필수 키 존재 여부)

3. **데이터셋 로딩 기능 브랜치 (`feature/dataset`)**
   - `tests/test_01_dataset.py` 작성: 데이터셋 초기화, 출력 형식 테스트
   - `src/anomaly_detection/datasets.py` 구현: AnomalyDataset 기본 클래스 및 MVTec/ViSA/BTAD 지원
   - 전처리(transforms) 통합 및 테스트
   - 데이터셋 분할(train/val/test) 기능 구현 및 검증

4. **백본 모델 로드 기능 브랜치 (`feature/backbone-loader`)**
   - `tests/test_backbone.py` 작성: 모델 이름, 가중치 로드, 출력 특징 맵 테스트
   - `src/anomaly_detection/backbone.py` 구현: ResNet, EfficientNet 등 지원
   - `return_layers` 기반 다중 특징 맵 추출 기능 구현

5. **모델 공통 인터페이스 브랜치 (`feature/model-base`)**
   - `tests/test_model_base.py` 작성: 공통 메서드(train_step, predict, save, load) 테스트
   - `src/anomaly_detection/models/base.py` 구현: 추상 기본 클래스 정의
   - 모델 등록 및 팩토리 패턴 준비 (필요 시)

6. **STFPM 모델 브랜치 (`feature/stfpm`)**
   - `tests/test_stfpm.py` 작성: 학생/교사 네트워크, 특징 차이, 이상 맵 생성 테스트
   - `src/anomaly_detection/models/stfpm.py` 구현: STFPM 아키텍처 및 손실 계산
   - 특징 피라미드 정합 동작 검증

7. **EfficientAD 모델 브랜치 (`feature/efficientad`)**
   - `tests/test_efficientad.py` 작성: 자율형 인코더, 재구성, 이상 점수 테스트
   - `src/anomaly_detection/models/efficientad.py` 구현: EfficientNet 기반 구조 통합
   - 사전 학습된 컴포넌트 로드 및 동기화 검증

8. **학습 엔진 브랜치 (`feature/trainer`)**
   - `tests/test_trainer.py` 작성: 학습 루프, 검증 주기, 체크포인트 저장 테스트
   - `src/anomaly_detection/trainer.py` 구현: Trainer 클래스 완성
   - 옵티마이저, 스케줄러, early stopping 통합

9. **실험 스크립트 브랜치 (`feature/experiments`)**
   - `experiments/train_stfpm.py` 작성 및 검증
   - `experiments/eval_efficientad.py` 작성 및 검증
   - 명령행 인자 처리 및 설정 병합 기능 구현

10. **통합 및 CI 브랜치 (`feature/ci-integration`)**
    - 전체 테스트 실행: `pytest tests/ --cov=src`
    - `.github/workflows/test.yml` 작성 및 CI 동작 검증
    - 코드 포맷팅 도구 설정 (black, flake8 등)

11. **문서화 브랜치 (`feature/docs`)**
    - `README.md` 완성: 설치, 실행, 예제 명령어 포함
    - `docs/` 내 문서 작성: 폴더 구조, 개발 가이드, 모델 추가 방법
    - 노트북 예제 검증 (`notebooks/` 내 파일)


### Git 명령어 모음

**1. 초기 설정**
```
cd projects/anomaly_detection
git init
git config --global init.defaultBranch main
git config --global core.quotepath false
git config --global core.autocrlf true
git config --global core.editor "code --wait"

git remote add origin https://github.com/your-username/anomaly_detection.git
git checkout -b main

git add .
git commit -m "chore: initialize project structure"
git push -u origin main

git add .gitignore
git commit -m "chore: add .gitignore"

git add pytest.ini
git commit -m "chore: configure pytest"

git add configs/paths.yaml
git commit -m "chore: define project paths"

git push
```

**2. 브랜치 생성 및 기능 개발** (테스트 - 구현 - 디버그 - 리팩토링)
```
git checkout main
git pull origin main
git checkout -b feature/<topic>

# 1. 테스트 (최초 -u)
git add tests/test_<topic>.py
git commit -m "test: add test_<topic>.py for <기능 설명>"
git push -u origin feature/<topic>

# 2. 구현
git add src/anomaly_detection/<module>.py
git commit -m "feat: implement <기능> for <topic>"
git push

# 3. 버그 수정 (필요 시)
git add src/anomaly_detection/<module>.py
git commit -m "debug: correct <문제점> in <기능>"
git push

# 4. 리팩터링 (필요 시)
git add src/anomaly_detection/<module>.py tests/test_<topic>.py
git commit -m "refactor: improve structure of <기능> for reusability"
git push
```

**3. 브랜치 작업 완료 및 병합**
```
git checkout main
git pull origin main

git branch -d feature/<이름>
git push origin --delete feature/<이름>
```
