# src/defectvad/data/transforms.py

import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_train_transform(img_size=256, crop_size=None, normalize=True, mean=MEAN, std=STD):
    transforms = [T.Resize((img_size, img_size))]
    if crop_size is not None:
        transforms.append(T.CenterCrop((crop_size, crop_size)))
    transforms.append(T.ToTensor())
    if normalize:
        transforms.append(T.Normalize(mean=MEAN, std=STD))
    return T.Compose(transforms)


def get_test_transform(img_size=256, crop_size=None, normalize=True, mean=MEAN, std=STD):
    transforms = [T.Resize((img_size, img_size))]
    if crop_size is not None:
        transforms.append(T.CenterCrop((crop_size, crop_size)))
    transforms.append(T.ToTensor())
    if normalize:
        transforms.append(T.Normalize(mean=MEAN, std=STD))
    return T.Compose(transforms)


def get_mask_transform(img_size=256, crop_size=None):
    transforms = [T.Resize((img_size, img_size))]
    if crop_size is not None:
        transforms.append(T.CenterCrop((crop_size, crop_size)))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
