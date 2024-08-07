import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class ReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples, self.label_to_int = self._load_samples()

    def _load_samples(self):
        samples = []
        label_to_int = {}
        current_label = 0
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                if class_dir not in label_to_int:
                    label_to_int[class_dir] = current_label
                    current_label += 1
                label_int = label_to_int[class_dir]
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    samples.append((img_path, label_int))
        return samples, label_to_int

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_samples_from_dir(root_dir):
    samples = []
    label_to_int = {}
    current_label = 0
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            if class_dir not in label_to_int:
                label_to_int[class_dir] = current_label
                current_label += 1
            label_int = label_to_int[class_dir]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                samples.append((img_path, label_int))
    return samples

def inspect_data_distribution(root_dir):
    samples = get_samples_from_dir(root_dir)
    labels = [label for _, label in samples]
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"Number of unique classes: {len(unique_labels)}")
    print(f"Total number of images: {len(samples)}")
    print(f"Average images per class: {np.mean(counts):.2f}")

    plt.figure(figsize=(10, 5))
    plt.bar(unique_labels, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Data Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.show()

def get_reid_dataloaders(root_dir, batch_size, transform, test_size=0.2):
    dataset = ReIDDataset(root_dir, transform=transform)
    val_size = int(len(dataset) * test_size)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
