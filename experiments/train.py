import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Define the LabelRemapper class
class LabelRemapper:
    def __init__(self, original_labels):
        unique_labels = sorted(set(original_labels))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
    
    def remap(self, label):
        return self.label_map[label]

# Function to remap dataset labels
def remap_dataset_labels(dataset):
    all_labels = [label for _, label in dataset.samples]
    remapper = LabelRemapper(all_labels)
    
    new_samples = [(path, remapper.remap(label)) for path, label in dataset.samples]
    dataset.samples = new_samples
    dataset.targets = [label for _, label in new_samples]
    
    return len(remapper.label_map)

# Define the train_epoch function
def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Debugging: Print out labels to check their range
        invalid_labels = labels[(labels < 0) | (labels >= num_classes)]
        if len(invalid_labels) > 0:
            print(f"Found invalid labels in the batch: {invalid_labels}")
            # Find corresponding image paths for the invalid labels
            invalid_indices = torch.nonzero((labels < 0) | (labels >= num_classes), as_tuple=True)[0]
            invalid_images = [dataloader.dataset.samples[idx][0] for idx in invalid_indices]
            print(f"Corresponding image paths: {invalid_images}")
            raise ValueError(f"Labels are out of valid range [0, {num_classes-1}]")

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc

# Define the train_model function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, save_dir=None):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    num_classes = len(set([label for _, label in train_loader.dataset.samples]))

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Train phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        val_loss = running_loss / total_samples
        val_acc = running_corrects.double() / total_samples

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            if save_dir:
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(best_model_wts, checkpoint_path)
                print(f'Saved best model checkpoint: {checkpoint_path}')

        scheduler.step()

    model.load_state_dict(best_model_wts)
    return model, train_losses, train_accs, val_losses, val_accs

# Main script to set up data, model, and training
def main():
    data_dir = "data"
    save_dir = "checkpoints"  # Define the directory to save checkpoints
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    
    # Remap labels and get the number of classes
    num_classes = remap_dataset_labels(image_datasets['train'])
    remap_dataset_labels(image_datasets['val'])

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, train_losses, train_accs, val_losses, val_accs = train_model(
        model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, device, num_epochs, save_dir
    )

if __name__ == '__main__':
    main()
