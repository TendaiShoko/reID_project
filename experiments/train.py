# experiments/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
import os
import numpy as np
from utils.visualization import plot_training_history
from utils.metrics import compute_accuracy, compute_mAP, compute_cmc

# Set up logging
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    best_val_map = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 10)

        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, val_acc, val_map = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_map)
        else:
            scheduler.step()

        # Log results
        logger.info(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} mAP: {val_map:.4f}')

        # Save results for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'New best model saved with mAP: {best_val_map:.4f}')

        # Print sample predictions every 5 epochs
        if epoch % 5 == 0:
            print_sample_predictions(model, train_loader, device, "Training")
            print_sample_predictions(model, val_loader, device, "Validation")

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                          save_path=os.path.join(save_dir, 'training_history.png'))
    logger.info("Training completed.")
    return model, train_losses, train_accs, val_losses, val_accs

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.float() / len(dataloader.dataset)

    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu())
            all_outputs.extend(outputs.cpu())

    all_labels = torch.stack(all_labels)
    all_outputs = torch.stack(all_outputs)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = compute_accuracy(all_outputs, all_labels)
    epoch_map = compute_mAP(all_outputs.numpy(), all_labels.numpy())

    return epoch_loss, epoch_acc, epoch_map

def print_sample_predictions(model, dataloader, device, phase):
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(min(5, len(inputs))):
                logger.info(f"{phase} - True: {labels[i].item()}, Predicted: {preds[i].item()}")
            break

    logger.info(f"{phase} - Unique predicted classes: {set(preds.cpu().numpy())}")

if __name__ == "__main__":
    # This block allows you to run some tests directly on this script if needed
    pass
