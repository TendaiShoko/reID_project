import logging
import torch
import yaml
from data.datasets import get_reid_dataloaders, inspect_data_distribution
from data.transforms import get_transform
from models.reid_model import ReIDModel
from experiments.train import train_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def inspect_data_loading(dataloader):
    for inputs, labels in dataloader:
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels: {labels}")
        break  # Only inspect the first batch


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Inspect data distribution
    inspect_data_distribution(data_dir)

    # Load data
    train_loader, val_loader = get_reid_dataloaders(data_dir, batch_size, get_transform(is_train=True), test_size=0.2)

    logger.info(f"Number of training samples: {len(train_loader.dataset)}")
    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")

    # Inspect data loading
    inspect_data_loading(train_loader)
    inspect_data_loading(val_loader)

    # Initialize model
    num_classes = len(set([label for _, label in train_loader.dataset.dataset.samples]))
    model = ReIDModel(num_classes).to(device)
    logger.info(f"Number of classes: {num_classes}")

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                num_epochs=num_epochs, device=device)

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
