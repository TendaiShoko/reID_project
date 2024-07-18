import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class ReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        if len(self.samples) == 0:
            raise ValueError("No samples found. Check the directory structure and paths.")

    def _load_samples(self):
        samples = []
        img_dirs = ['image_train', 'image_test']
        for img_dir in img_dirs:
            full_img_dir = os.path.join(self.root_dir, img_dir)
            if os.path.isdir(full_img_dir):
                for img_name in os.listdir(full_img_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
                        img_path = os.path.join(full_img_dir, img_name)
                        id = int(img_name.split('_')[0])  # Assuming ID is the first part of the file name
                        samples.append((img_path, id))
        print(f"Loaded {len(samples)} samples from {self.root_dir}")
        if len(samples) == 0:
            print("No samples found. Ensure your dataset directory structure is correct.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_samples_from_dir(root_dir):
    samples = []
    img_dirs = ['image_train', 'image_test']
    for img_dir in img_dirs:
        full_img_dir = os.path.join(root_dir, img_dir)
        if os.path.isdir(full_img_dir):
            for img_name in os.listdir(full_img_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
                    img_path = os.path.join(full_img_dir, img_name)
                    id = int(img_name.split('_')[0])  # Assuming ID is the first part of the file name
                    samples.append((img_path, id))
    return samples

def inspect_data_distribution(root_dir):
    samples = get_samples_from_dir(root_dir)
    labels = [label for _, label in samples]
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique_labels, counts)
    plt.title('Data Distribution')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.savefig('data_distribution.png')
    print("Data distribution plot saved as 'data_distribution.png'")

def get_reid_dataloaders(root_dir, batch_size, transform, test_size=0.2):
    dataset = ReIDDataset(root_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check the directory structure and paths.")
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
