import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


def plot_embeddings(embeddings, labels, output_path='embeddings_tsne.png'):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of embeddings')
    plt.savefig(output_path)
    plt.close()


def plot_metrics(metric_values, labels, metric_name, output_path):
    plt.figure(figsize=(10, 6))
    for i, values in enumerate(metric_values):
        plt.plot(values, label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def generate_comparison_table(data, output_path):
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(output_path, index=False)
    print(f'Table saved to {output_path}')

# Example usage:
# plot_metrics([train_accuracies, val_accuracies], ['Train', 'Validation'], 'Accuracy', 'accuracy_over_epochs.png')
# generate_comparison_table({
#     'Model': ['MoE', 'KD'],
#     'Rank-1': [0.85, 0.82],
#     'Rank-5': [0.92, 0.90],
#     'Rank-10': [0.94, 0.93],
#     'mAP': [0.75, 0.73]
# }, 'comparison_table.csv')
