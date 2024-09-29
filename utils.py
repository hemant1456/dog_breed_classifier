# download the file using kaggle api kaggle datasets download -d khushikhushikhushi/dog-breed-image-dataset

import kaggle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


# use local kaggle.json file to authenticate

def download_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('khushikhushikhushi/dog-breed-image-dataset', path='.', unzip=True)


# create a function to split the dataset into train, val and test
def split_dataset(cfg):
    # read dataset folder into dataset
    # basic transforms
    basic_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(cfg.folder.dataset, transform=basic_transforms)
    # split the dataset into train, val and test
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [cfg.split_ratio.train, cfg.split_ratio.val, cfg.split_ratio.test])
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Val dataset: {len(val_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

# create a function to create dataloader from the dataset

def create_dataloader(cfg):
    # create a dataloader from the dataset
    train_dataset, val_dataset, test_dataset = split_dataset(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataloader.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.dataloader.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.dataloader.batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader