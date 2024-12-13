import logging
import torch
import yaml
from data.datasets import get_reid_dataloaders, inspect_data_distribution
from data.transforms import get_transform
from models.reid_model import ReIDModel
from experiments.train import train_model
from experiments.evaluate import evaluate_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset(dataset):
    from collections import Counter
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    print(f"Number of unique classes: {len(class_counts)}")
    print(f"Total number of images: {len(labels)}")
    print(f"Class distribution: {class_counts}")
    print(f"Min class size: {min(class_counts.values())}")
    print(f"Max class size: {max(class_counts.values())}")

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    save_dir = config['save_dir']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Inspect data distribution
    inspect_data_distribution(data_dir)

    # Load data
    train_loader, val_loader = get_reid_dataloaders(data_dir, batch_size, get_transform(is_train=True), test_size=0.2)

    logger.info(f"Number of training samples: {len(train_loader.dataset)}")
    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")

    # Convert DataLoader to dataset
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    # Analyze datasets
    print("Analyzing training dataset...")
    analyze_dataset(train_dataset)
    
    print("Analyzing validation dataset...")
    analyze_dataset(val_dataset)

    # Initialize model
    num_classes = len(set([label for _, label in train_loader.dataset.dataset.samples]))
    model = ReIDModel(num_classes).to(device)
    logger.info(f"Number of classes: {num_classes}")

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # Train the model
    trained_model, train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs, device=device, save_dir=save_dir
    )

    logger.info("Training completed.")

    # Evaluate the model
    logger.info("Evaluating model on validation set...")
    evaluate_model(model, val_loader, device, save_dir)


if __name__ == "__main__":
    main()
