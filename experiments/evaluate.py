import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import compute_cmc, compute_map
from utils.visualization import plot_embeddings, generate_comparison_table


def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(batch_labels.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def compute_distance_matrix(query_features, gallery_features):
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    dist_mat = 2 - 2 * np.dot(query_features, gallery_features.T)
    return dist_mat


def evaluate(model, query_loader, gallery_loader, device, output_prefix=''):
    query_features, query_labels = extract_features(model, query_loader, device)
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device)

    dist_mat = compute_distance_matrix(query_features, gallery_features)

    cmc_scores = compute_cmc(dist_mat, query_labels, gallery_labels)
    mAP = compute_map(dist_mat, query_labels, gallery_labels)

    print(f"Rank-1: {cmc_scores[0]:.4f}")
    print(f"Rank-5: {cmc_scores[4]:.4f}")
    print(f"Rank-10: {cmc_scores[9]:.4f}")
    print(f"mAP: {mAP:.4f}")

    # Visualize embeddings
    all_features = np.concatenate([query_features, gallery_features], axis=0)
    all_labels = np.concatenate([query_labels, gallery_labels], axis=0)
    plot_embeddings(all_features, all_labels, output_path=f'{output_prefix}_embeddings_tsne.png')

    # Generate comparison table
    data = {
        'Model': [output_prefix],
        'Rank-1': [cmc_scores[0]],
        'Rank-5': [cmc_scores[4]],
        'Rank-10': [cmc_scores[9]],
        'mAP': [mAP]
    }
    generate_comparison_table(data, output_path=f'{output_prefix}_comparison_table.csv')

    return cmc_scores, mAP
