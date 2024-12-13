import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_embeddings(features, labels, num_classes, save_path='embeddings_tsne.png'):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    palette = sns.color_palette("hsv", num_classes)
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        palette=palette,
        legend="full",
        alpha=0.3
    )
    plt.title('T-SNE of Features')
    plt.savefig(save_path)
    plt.show()
