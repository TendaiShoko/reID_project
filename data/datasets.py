import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class ReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for folder in ['image_train', 'image_test', 'image_query']:
            folder_path = os.path.join(self.root_dir, folder)
            for img_name in os.listdir(folder_path):
                if img_name.endswith('.jpg'):
                    class_id = img_name.split('_')[0]  # Assuming the class id is the first part of the filename
                    img_path = os.path.join(folder_path, img_name)
                    samples.append((img_path, int(class_id)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_reid_dataloaders(root_dir, batch_size, transform, test_size=0.2, num_workers=4):
    full_dataset = ReIDDataset(root_dir, transform=transform)
    train_size = int((1 - test_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def inspect_data_distribution(root_dir):
    samples = []
    for folder in ['image_train', 'image_test', 'image_query']:
        folder_path = os.path.join(root_dir, folder)
        for img_name in os.listdir(folder_path):
            if img_name.endswith('.jpg'):
                class_id = img_name.split('_')[0]
                img_path = os.path.join(folder_path, img_name)
                samples.append((img_path, int(class_id)))

    unique_labels = set([label for _, label in samples])
    num_classes = len(unique_labels)
    num_images = len(samples)
    average_images_per_class = num_images / num_classes

    print(f"Number of unique classes: {num_classes}")
    print(f"Total number of images: {num_images}")
    print(f"Average images per class: {average_images_per_class:.2f}")

    import matplotlib.pyplot as plt
    from collections import Counter
    labels = [label for _, label in samples]
    class_counts = Counter(labels)

    plt.figure(figsize=(10, 5))
    plt.hist(class_counts.values(), bins=50)
    plt.xlabel('Number of images per class')
    plt.ylabel('Number of classes')
    plt.title('Class Distribution')
    plt.show()
