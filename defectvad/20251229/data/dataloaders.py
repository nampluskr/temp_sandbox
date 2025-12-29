# sec/defectvad/data/dataloaders.py
import torch
from torch.utils.data import DataLoader


def get_train_loader(dataset, batch_size=16, collate_fn=None, **kwargs):
    config = {
        "num_workers": 8,
        "pin_memory": True,
        "shuffle": True,
        "drop_last": True,
    }
    config.update(kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **config
    )


def get_test_loader(dataset, batch_size=16, collate_fn=None, **kwargs):
    config = {
        "num_workers": 8,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    }
    config.update(kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **config
    )

def get_dataloader(dataset, batch_size=16, collate_fn=None, **kwargs):
    config = {
        "num_workers": 8,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    }
    config.update(kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **config
    )
