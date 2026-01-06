# data/datasets.py

from abc import ABC, abstractmethod
import os
from glob import glob
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as T


class BaseDataset(Dataset, ABC):
    DATASET = ""    # dataset name: mvtec | visa | btad
    CATEGORIES = []

    def __init__(self, root_dir, category, split, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.category = category
        self.category_dir = os.path.join(root_dir, category)
        self.split = split
        self.transform = transform or T.ToTensor()
        self.mask_transform = mask_transform or T.ToTensor()
        self.samples = []

        if split == "train":
            self._load_train_samples()
        elif split == "test":
            self._load_test_samples()
        else:
            raise ValueError(f"split must be 'train' or 'test': {split}")

    @abstractmethod
    def _load_train_samples(self):
        raise NotImplementedError

    @abstractmethod
    def _load_test_samples(self):
        raise NotImplementedError

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        self.height = image.shape[1]
        self.width = image.shape[2]
        return image

    def _load_mask(self, mask_path):
        if mask_path is None:
            return torch.zeros((1, self.height, self.width))
        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return mask

    def count_category(self, category):
        return sum(sample["label"] == 0 for sample in self.samples)

    def count_normal(self, category=None):
        if category is None:
            return sum(sample["label"] == 0 for sample in self.samples)
        else:
            return sum(sample["label"] == 0 and sample["category"] == category 
                for sample in self.samples)


    def count_anomaly(self, category=None):
        if category is None:
            return sum(sample["label"] == 1 for sample in self.samples)
        else:
            return sum(sample["label"] == 1 and sample["category"] == category 
                for sample in self.samples)

    def filter(self, category):
        if isinstance(category, (list, tuple, set)):
            indices = [i for i in range(len(self)) if self.categories[i] in category]
        else:
            indices = [i for i in range(len(self)) if self.categories[i] == category]
        return Subset(self, indices)

    @classmethod
    def concat(cls, root_dir, categories, split, transform=None, mask_transform=None):
        datasets = []
        for category in categories:
            if category not in cls.CATEGORIES:
                raise ValueError(f"Unknown category: {category}. Available: {cls.CATEGORIES}")

            dataset = cls(
                root_dir=root_dir,
                category=category,
                split=split,
                transform=transform,
                mask_transform=mask_transform
            )
            datasets.append(dataset)
        return ConcatDataset(datasets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image": self._load_image(sample["image_path"]),
            "label": torch.tensor(sample["label"]).long(),
            "defect_type": sample["defect_type"],
            "mask": self._load_mask(sample["mask_path"]),
            "dataset": self.DATASET,
            "category": self.category,
            "filename": os.path.basename(sample['image_path']),
        }


# =========================================================
# MVTec
# =========================================================

class MVTecDataset(BaseDataset):
    DATASET = "mvtec"
    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def _load_train_samples(self):
        normal_dir = os.path.join(self.category_dir, "train", "good")
        for image_path in sorted(glob(os.path.join(normal_dir, "*.png"))):
            self.samples.append({
                "image_path": image_path,
                "label": 0,
                "defect_type": "normal",
                "mask_path": None
            })

    def _load_test_samples(self):
        test_dir = os.path.join(self.category_dir, "test")
        mask_dir = os.path.join(self.category_dir, "ground_truth")

        for defect_type in sorted(os.listdir(test_dir)):
            for image_path in sorted(glob(os.path.join(test_dir, defect_type, "*.png"))):

                if defect_type == "good":
                    self.samples.append({
                        "image_path": image_path,
                        "label": 0,
                        "defect_type": "normal",
                        "mask_path": None
                    })
                else:
                    image_name = os.path.basename(image_path)
                    mask_name = os.path.splitext(image_name)[0] + "_mask.png"
                    mask_path = os.path.join(mask_dir, defect_type, mask_name)

                    self.samples.append({
                        "image_path": image_path,
                        "label": 1,
                        "defect_type": defect_type,
                        "mask_path": mask_path
                    })


# =========================================================
# ViSA
# =========================================================

class ViSADataset(BaseDataset):
    DATASET = "visa"
    CATEGORIES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum'
    ]

    def __init__(self, root_dir, category, split, transform=None, mask_transform=None):
        csv_path = os.path.join(root_dir, "split_csv", "1cls.csv")
        df = pd.read_csv(csv_path)
        self.df = df[df["object"] == category].reset_index(drop=True)
        self.root_dir = root_dir

        super().__init__(root_dir, category, split, transform, mask_transform)

    def _load_train_samples(self):
        df = self.df[self.df["split"] == "train"].reset_index(drop=True)
        self._load_samples_from_df(df)

    def _load_test_samples(self):
        df = self.df[self.df["split"] == "test"].reset_index(drop=True)
        self._load_samples_from_df(df)

    def _load_samples_from_df(self, df):
        image_paths = [os.path.join(self.root_dir, path) for path in df["image"]]
        mask_paths = [
            os.path.join(self.root_dir, path) if pd.notna(path) else None
            for path in df["mask"]
        ]
        labels = (df["label"] != "normal").astype(int).tolist()
        defect_types = df["label"].tolist()

        self.samples = [
            {
                "image_path": image_path,
                "label": label,
                "defect_type": defect_type,
                "mask_path": mask_path
            } 
            for image_path, label, defect_type, mask_path 
            in zip(image_paths, labels, defect_types, mask_paths)
        ]


# =========================================================
# BTAD
# =========================================================

class BTADDataset(BaseDataset):
    DATASET = "btad"
    CATEGORIES = ['01', '02', '03']

    def _load_train_samples(self):
        normal_dir = os.path.join(self.category_dir, "train", "ok")
        for image_path in sorted(glob(os.path.join(normal_dir, "*.*"))):
            # ext = os.path.splitext(image_path)[1].lower()
            # if ext in ("png", "jpg", "bmp"):
            self.samples.append({
                "image_path": image_path,
                "label": 0,
                "defect_type": "normal",
                "mask_path": None
            })

    def _load_test_samples(self):
        normal_dir = os.path.join(self.category_dir, "test", "ok")
        anomaly_dir = os.path.join(self.category_dir, "test", "ko")
        mask_dir = os.path.join(self.category_dir, "ground_truth", "ko")

        for image_path in sorted(glob(os.path.join(normal_dir, "*.*"))):
            # ext = os.path.splitext(image_path)[1].lower()
            # if ext in ("png", "jpg", "bmp"):
            self.samples.append({
                "image_path": image_path,
                "label": 0,
                "defect_type": "normal",
                "mask_path": None
            })

        for image_path in sorted(glob(os.path.join(anomaly_dir, "*.*"))):
            # ext = os.path.splitext(image_path)[1].lower()
            # if ext in ("png", "jpg", "bmp"):
            image_name = os.path.basename(image_path)
            mask_name = os.path.splitext(image_name)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)

            self.samples.append({
                "image_path": image_path,
                "label": 1,
                "defect_type": "anomaly",
                "mask_path": mask_path
            })
