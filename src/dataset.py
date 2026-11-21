
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Indoor67Dataset(Dataset):
    def __init__(self, dataframe, image_root, class_to_idx=None,
                 transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

        if "label" in self.df.columns:
            if class_to_idx is None:
                classes = sorted(self.df["label"].unique())
                class_to_idx = {
                    name: idx for idx, name in enumerate(classes)
                }

        self.class_to_idx = class_to_idx
        self.classes = (
            sorted(class_to_idx, key=class_to_idx.get)
            if class_to_idx else []
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["image_path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        if "label" in self.df.columns:
            label = self.class_to_idx[row["label"]]
            return image, label

        return image, row["image_id"]


def read_indoor_csv(csv_path, is_test=False):
    columns = (
        ["image_id", "image_path"]
        if is_test
        else ["image_id", "image_path", "label"]
    )

    # Works with either headerless CSVs or CSVs with
    # the expected column names.
    df = pd.read_csv(csv_path, header=None)

    if df.iloc[0].tolist() == columns:
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] != len(columns):
        raise ValueError(
            f"Expected {len(columns)} columns, "
            f"found {df.shape[1]} in {csv_path}"
        )

    df.columns = columns
    return df


def get_transforms(image_size=224, augment=False):
    operations = [transforms.Resize((image_size, image_size))]

    if augment:
        operations.append(transforms.RandomHorizontalFlip())

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(operations)


def create_dataloaders(
    data_dir="data",
    batch_size=32,
    image_size=224,
    seed=42,
    val_size=0.2
):
    data_dir = Path(data_dir)
    train_df = read_indoor_csv(data_dir / "train.csv")

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label"]
    )

    class_names = sorted(train_df["label"].unique())
    class_to_idx = {
        name: idx for idx, name in enumerate(class_names)
    }

    train_dataset = Indoor67Dataset(
        train_split,
        data_dir / "Images",
        class_to_idx=class_to_idx,
        transform=get_transforms(image_size, augment=True)
    )

    val_dataset = Indoor67Dataset(
        val_split,
        data_dir / "Images",
        class_to_idx=class_to_idx,
        transform=get_transforms(image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, class_to_idx
