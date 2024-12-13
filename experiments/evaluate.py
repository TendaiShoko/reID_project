# experiments/evaluate.py
import torch
from tqdm import tqdm
import logging
from utils.metrics import compute_mAP, compute_cmc

# Set up logging
logger = logging.getLogger(__name__)

def evaluate_model(model, query_loader, gallery_loader, device, save_dir):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        # Process query set
        for inputs, labels in tqdm(query_loader, desc='Processing query set'):
            inputs = inputs.to(device)
            features = model(inputs)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        query_features = torch.cat(all_features)
        query_labels = torch.cat(all_labels)

        # Reset for gallery set
        all_features = []
        all_labels = []

        # Process gallery set
        for inputs, labels in tqdm(gallery_loader, desc='Processing gallery set'):
            inputs = inputs.to(device)
            features = model(inputs)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        gallery_features = torch.cat(all_features)
        gallery_labels = torch.cat(all_labels)

    # Compute metrics
    mAP = compute_mAP(query_features, gallery_features, query_labels, gallery_labels)
    cmc_scores = compute_cmc(query_features, gallery_features, query_labels, gallery_labels)

    logger.info(f'mAP: {mAP:.4f}')
    logger.info(f'CMC Scores: Rank-1: {cmc_scores[0]:.4f}, Rank-5: {cmc_scores[4]:.4f}, Rank-10: {cmc_scores[9]:.4f}')

    # Save evaluation results
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f'mAP: {mAP:.4f}\n')
        f.write(f'CMC Scores: Rank-1: {cmc_scores[0]:.4f}, Rank-5: {cmc_scores[4]:.4f}, Rank-10: {cmc_scores[9]:.4f}\n')

    return mAP, cmc_scores
