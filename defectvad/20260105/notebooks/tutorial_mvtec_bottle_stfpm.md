## MVTec / bottle / STFPM (10 epochs)

### Setup

```python
import os
import sys

PROJECT_DIR = "d:\\Non_Documents\\_github\\defectvad"
SOURCE_DIR = os.path.join(PROJECT_DIR, "src")

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)
    
os.environ["BACKBONE_DIR"] = "d:\\Non_Documents\\backbones"
os.environ["DATASET_DIR"] = "e:\\datasets"
```

### Hyperparameters

```python
# Dataset
DATASET_DIR = "e:\\datasets\\mvtec"
CATEGORY = "bottle"
IMG_SIZE = 256
CROP_SIZE = None
NORMALIZE = True

# Train Dataloader
TRAIN_BATCH_SIZE = 16
TRAIN_SHUFFLE = True
TRAIN_DROP_LAST = True
TRAIN_NUM_WORKERS = 0
TRAIN_PIN_MEMORY = False

# Test Dataloader
TEST_BATCH_SIZE = 1
TEST_SHUFFLE = False
TEST_DROP_LAST = False
TEST_NUM_WORKERS = 0
TEST_PIN_MEMORY = False

# Trainer
MAX_EPOCHS = 10
```

### Datasets

```python
from defectvad.data.datasets import MVTecDataset
from defectvad.data.transforms import get_image_transform, get_mask_transform

train_dataset = MVTecDataset(
    root_dir=DATASET_DIR,
    category=CATEGORY,
    split="train",
    transform=get_image_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
    mask_transform=get_mask_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE)
)

test_dataset = MVTecDataset(
    root_dir=DATASET_DIR,
    category=CATEGORY,
    split="test",
    transform=get_image_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE, normalize=NORMALIZE),
    mask_transform=get_mask_transform(img_size=IMG_SIZE, crop_size=CROP_SIZE)
)

len(train_dataset), len(test_dataset)
```

### Dataloaders

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=TRAIN_SHUFFLE,
    drop_last=TRAIN_DROP_LAST,
    num_workers=TRAIN_NUM_WORKERS,
    pin_memory=TRAIN_PIN_MEMORY,
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=TEST_BATCH_SIZE,
    shuffle=TEST_SHUFFLE,
    drop_last=TEST_DROP_LAST,
    num_workers=TEST_NUM_WORKERS,
    pin_memory=TEST_PIN_MEMORY,
)

len(train_loader), len(test_loader)
```

### Model / Trainer
```python
from defectvad.models.stfpm.model_trainer import STFPM, STFPMTrainer

model = STFPM(backbone="resnet50", layers=["layer1", "layer2", "layer3"])
trainer = STFPMTrainer(model)

trainer.fit(train_loader, max_epochs=MAX_EPOCHS)
```

### Evaluator

```python
from defectvad.common.evaluator import Evaluator

evaluator = Evaluator(model)

image_results = evaluator.evaluate_image_level(test_loader)
print(" > Image: " + ", ".join([f"{k}: {v:.4f}" for k, v in image_results.items()]))

pixel_results = evaluator.evaluate_pixel_level(test_loader)
print(" > Pixel: " + ", ".join([f"{k}: {v:.4f}" for k, v in pixel_results.items()]))
```

### Visualizer

```python
from defectvad.common.visualizer import Visualizer

preds = model.predict(test_loader)
visualizer = Visualizer(preds)

visualizer.show_anomaly(max_samples=1)
visualizer.show_normal(max_samples=1)
```
