"""DeepWeeds loading, split validation, transforms, and DataLoader helpers."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # allows metadata-only tools to import the module outside Kaggle
    class _TorchDataset:
        pass

NUM_CLASSES = 9
EXPECTED_IMAGES = 17_509
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read train/val/test CSVs; split metadata only needs Filename and Label."""
    if not 0 <= int(fold) <= 4:
        raise ValueError("fold must be between 0 and 4")
    labels_dir = Path(labels_dir)
    frames = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{int(fold)}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {path}")
        frame = pd.read_csv(path)
        missing = {"Filename", "Label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Validate fold integrity and image availability; return counts for the report."""
    images_dir = Path(images_dir)
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {}
    counts = {}
    per_class = {}
    for split, frame in frames.items():
        missing_cols = {"Filename", "Label"} - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{split} CSV is missing columns: {sorted(missing_cols)}")
        if frame["Filename"].isna().any() or frame["Label"].isna().any():
            raise ValueError(f"{split} CSV has empty Filename or Label values")
        filenames = frame["Filename"].astype(str)
        duplicates = filenames[filenames.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(f"{split} contains duplicate filenames: {duplicates[:5]}")
        numeric_labels = pd.to_numeric(frame["Label"], errors="raise")
        if (not numeric_labels.mod(1).eq(0).all()
                or not numeric_labels.between(0, NUM_CLASSES - 1).all()):
            invalid = numeric_labels[~numeric_labels.between(0, NUM_CLASSES - 1)
                                     | ~numeric_labels.mod(1).eq(0)].unique().tolist()
            raise ValueError(f"{split} has invalid labels outside integer range 0..{NUM_CLASSES - 1}: {invalid[:5]}")
        labels = numeric_labels.astype(int)
        names[split] = set(filenames)
        counts[split] = int(len(frame))
        class_counts = labels.value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        per_class[split] = {CLASS_NAMES[i]: int(class_counts.iloc[i]) for i in range(NUM_CLASSES)}

    overlaps = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    if any(overlaps.values()):
        nonempty = {key: value[:5] for key, value in overlaps.items() if value}
        raise ValueError(f"Fold has overlapping filenames: {nonempty}")
    total = len(names["train"] | names["val"] | names["test"])
    if total != EXPECTED_IMAGES:
        raise ValueError(f"Fold union has {total} images; expected {EXPECTED_IMAGES}")
    missing_images = [name for name in names["train"] | names["val"] | names["test"]
                      if not (images_dir / name).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} CSV images are absent from {images_dir}; examples: {missing_images[:5]}"
        )

    ratios = {key: counts[key] / total for key in counts}
    if any(abs(ratios[key] - target) > 0.01 for key, target in
           (("train", 0.6), ("val", 0.2), ("test", 0.2))):
        raise ValueError(f"Split proportions differ by more than 1 percentage point: {ratios}")
    if verbose:
        print(f"Split counts: {counts}; union={total}; overlaps=0")
        for split, values in per_class.items():
            print(f"{split} class counts: {values}")
    return {"n": counts, "per_class": per_class, "overlap": overlaps,
            "union": total, "ratios": ratios}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build deterministic validation transforms or one of the documented train policies."""
    from torchvision import transforms

    if img_size < 32:
        raise ValueError("img_size must be at least 32")
    if train:
        ops = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
        if aug == "color":
            ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
        elif aug == "trivial":
            augment = getattr(transforms, "TrivialAugmentWide", None)
            ops.append(augment() if augment else transforms.RandAugment())
        elif aug == "randaug":
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif aug != "basic":
            raise ValueError(f"Unknown augmentation: {aug}")
    else:
        # DeepWeeds images are 256x256. Resize then center crop, with no random ops.
        ops = [transforms.Resize(img_size + 32), transforms.CenterCrop(img_size)]
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class DeepWeedsDataset(_TorchDataset):
    """Small in-memory index over image paths; image pixels are decoded on demand."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not {"Filename", "Label"}.issubset(df.columns):
            raise ValueError("df must contain Filename and Label columns")
        self.filenames = df["Filename"].astype(str).tolist()
        numeric_labels = pd.to_numeric(df["Label"], errors="raise")
        if (numeric_labels.isna().any() or not numeric_labels.mod(1).eq(0).all()
                or not numeric_labels.between(0, NUM_CLASSES - 1).all()):
            raise ValueError(f"Label values must be integers in 0..{NUM_CLASSES - 1}")
        self.labels = numeric_labels.astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        from PIL import Image

        path = self.images_dir / self.filenames[i]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(self.labels[i]), self.filenames[i]


def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0):
    """Create a reproducible loader; evaluation preserves CSV row order."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sampler not in (None, "balanced"):
        raise ValueError(f"Unknown sampler: {sampler}")
    generator = torch.Generator().manual_seed(int(seed))
    dataset = DeepWeedsDataset(df, images_dir, transform)
    weighted_sampler = None
    if train and sampler == "balanced":
        labels = np.asarray(dataset.labels, dtype=np.int64)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        weights = np.asarray([1.0 / counts[label] for label in labels], dtype=np.float64)
        weighted_sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True,
            generator=generator,
        )

    # At most one queued batch per worker; no image cache and no persistent worker pool.
    worker_options = {"prefetch_factor": 1, "multiprocessing_context": "spawn"} if num_workers > 0 else {}
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=bool(train and weighted_sampler is None),
        sampler=weighted_sampler, num_workers=max(0, int(num_workers)),
        pin_memory=bool(torch.cuda.is_available()), drop_last=False,
        worker_init_fn=seed_worker, generator=generator,
        persistent_workers=False, **worker_options,
    )
