# defectvad/models/efficientad/model_config.yaml

#####################################################################
# Model
#####################################################################

model:
  name: efficientad
  module: defectvad.models.efficientad.torch_model
  class: EfficientAdModel

#####################################################################
# Trainer
#####################################################################

trainer:
  name: efficientad
  module: defectvad.models.efficientad.trainer
  class: EfficientAdTrainer

  params:
    teacher_out_channels: 384
    model_size: small          # small | medium
    padding: false
    pad_maps: true

  max_epochs: 10
  validate: true

#####################################################################
# Base
#####################################################################

seed: 42

path:
  backbone: /home/namu/myspace/NAMU/backbones
  dataset: /home/namu/myspace/NAMU/datasets
  mvtec: /home/namu/myspace/NAMU/datasets/mvtec
  visa: /home/namu/myspace/NAMU/datasets/visa
  btad: /home/namu/myspace/NAMU/datasets/btad

#####################################################################
# Dataset
#####################################################################

dataset:
  name: mvtec
  module: defectvad.data.datasets
  class: MVTecDataset
  # path: /home/namu/myspace/NAMU/datasets/mvtec
  # category: bottle

  img_size: 256
  crop_size: null
  normalize: false              # << INPORTANT !!!
  mean: [0.485, 0.456, 0.406]   # << No Effect !!!
  std: [0.229, 0.224, 0.225]    # << No Effect !!!

#####################################################################
# Dataloader
#####################################################################

dataloader:
  train:
    batch_size: 1         # << INPORTANT !!!

    params:
      shuffle: true
      drop_last: true
      num_workers: 8
      pin_memory: true

  test:
    batch_size: 1
    
    params:
      shuffle: false
      drop_last: false
      num_workers: 8
      pin_memory: true
  
