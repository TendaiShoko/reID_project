**reID_project**
*Overview*

This project aims to train a Re-Identification (ReID) model using the VeRi (Current training completed)  and Market1500 dataset. 
The model leverages a ResNet50 backbone, and the training process includes advanced techniques such as data augmentation, a combination of loss functions, gradient clipping, and learning rate scheduling.

**Directory Structure**


    reID_project/
    │
    ├── configs/
    │   ├── kd_config.yaml
    │   └── moe_config.yaml
    │
    ├── data/
    │   ├── Market_1501/
    │   │   ├── bounding_box_train/
    │   │   ├── bounding_box_test/
    │   │   ├── gt_bbox/
    │   │   ├── gt_query/
    │   │   ├── query/
    │   │   └── readme.txt
    │   └── VeRi/
    │       ├── image_query/
    │       ├── image_test/
    │       └── image_train/
    │
    ├── experiments/
    │   ├── __init__.py
    │   ├── train.py
    │
    ├── models/
    │   ├── __init__.py
    │   ├── foundation_models.py
    │   └── reid_model.py
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── metrics.py
    │   └── visualization.py
    │
    ├── venv/
    │
    ├── config.yaml
    ├── main.py
    └── requirements.txt

**Setup**
Install Dependencies

    pip install -r requirements.txt

**Configuration**
Create a config.yaml file with the following content:

    data_dir: '/Users/tendai/Desktop/reID_project/data/VeRi'
    batch_size: 32
    learning_rate: 0.001
    weight_decay: 0.0001
    num_epochs: 50

**Data Preparation**
Ensure your data directory is structured as follows:

    VeRi/
    ├── image_query/
    ├── image_test/
    └── image_train/

**Run the Project**
To train the model, run:


    python main.py

**Key Changes and Enhancements**

**1. Improved Model Architecture**
The model now uses a ResNet50 backbone with additional fully connected layers for better feature extraction and classification.

**2. Advanced Data Augmentation**
Utilized advanced data augmentation techniques to improve the model's robustness:


        from torchvision.transforms import autoaugment, transforms
        transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        autoaugment.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])

**3. Combination of Loss Functions**

    Implemented a combination of Cross-Entropy Loss and Triplet Loss:
        from torch.nn import TripletMarginLoss
        criterion_ce = nn.CrossEntropyLoss()
        criterion_triplet = TripletMarginLoss(margin=0.3)

**4. Learning Rate Scheduling**
Incorporated a learning rate scheduler with warm-up and decay:


       from torch.optim.lr_scheduler import OneCycleLR
    
       total_steps = len(train_loader) * config['num_epochs']
       scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.1)

**5. Gradient Clipping**
Added gradient clipping to prevent gradient explosions:

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
**6. Batch Hard Triplet Mining**
Implemented batch hard triplet mining for more effective training:


    def batch_hard_triplet_loss(labels, features, margin, squared=False):
    pairwise_dist = torch.cdist(features, features, p=2)
    
    mask_anchor_positive = labels.expand(len(labels), len(labels)).eq(labels.expand(len(labels), len(labels)).t())
    mask_anchor_negative = labels.expand(len(labels), len(labels)).ne(labels.expand(len(labels), len(labels)).t())
    
    anchor_positive_dist = mask_anchor_positive.float() * pairwise_dist
    hardest_positive_dist = anchor_positive_dist.max(1, keepdim=True)[0]
    
    max_anchor_negative_dist = pairwise_dist.max(1, keepdim=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
    hardest_negative_dist = anchor_negative_dist.min(1, keepdim=True)[0]
    
    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    triplet_loss = triplet_loss.mean()
    
    return triplet_loss
**7. Data Inspection**
Implemented a function to inspect data distribution:


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

**8. **Logging and Plotting****
Enhanced logging and added plotting of training history:

    def plot_training_history(train_losses, val_losses, train_accs, val_accs):
        import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    logger.info("Training history plot saved as 'training_history.png'")
