import json
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(augment=True):
    """Trả về (train_tf, eval_tf). augment=False dùng cho thí nghiệm ablation."""
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = eval_tf
    return train_tf, eval_tf


def _get_splits(targets):
    """Chia 70/15/15 có phân tầng theo lớp, lưu lại để mọi lần chạy dùng cùng một cách chia."""
    path = config.OUT_DIR / "splits.json"
    if path.exists():
        s = json.loads(path.read_text())
        if s["n"] == len(targets):
            return s["train"], s["val"], s["test"]
    idx = list(range(len(targets)))
    train_idx, rest = train_test_split(
        idx, test_size=0.30, stratify=targets, random_state=config.SEED)
    val_idx, test_idx = train_test_split(
        rest, test_size=0.50, stratify=[targets[i] for i in rest],
        random_state=config.SEED)
    path.write_text(json.dumps(
        {"n": len(targets), "train": train_idx, "val": val_idx, "test": test_idx}))
    return train_idx, val_idx, test_idx


def get_data(augment=True):
    train_tf, eval_tf = get_transforms(augment)
    train_full = datasets.ImageFolder(config.DATA_DIR, transform=train_tf)
    eval_full = datasets.ImageFolder(config.DATA_DIR, transform=eval_tf)
    classes = [config.TRANSLATE.get(c, c) for c in train_full.classes]
    targets = train_full.targets

    train_idx, val_idx, test_idx = _get_splits(targets)

    kw = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
              pin_memory=torch.cuda.is_available())
    loaders = {
        "train": DataLoader(Subset(train_full, train_idx), shuffle=True, **kw),
        "val": DataLoader(Subset(eval_full, val_idx), shuffle=False, **kw),
        "test": DataLoader(Subset(eval_full, test_idx), shuffle=False, **kw),
    }

    # Trọng số lớp (dùng khi dữ liệu mất cân bằng)
    counts = Counter(targets[i] for i in train_idx)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(classes) * counts[c]) for c in range(len(classes))],
        dtype=torch.float)

    return {
        "loaders": loaders,
        "classes": classes,
        "class_weights": weights,
        "counts": {classes[c]: counts[c] for c in range(len(classes))},
        "test_paths": [eval_full.samples[i][0] for i in test_idx],
        "test_targets": [targets[i] for i in test_idx],
    }


if __name__ == "__main__":
    d = get_data()
    print("Lớp:", d["classes"])
    print("Số ảnh train theo lớp:", d["counts"])
    for k, v in d["loaders"].items():
        print(k, len(v.dataset), "ảnh")
