import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


def compute_clone_metrics(distances, labels, threshold=None):
    """
    Evaluates clone detection performance based on the Joint Representation distances.

    Args:
        distances (list or np.array): Euclidean distances between H_A and H_B.
        labels (list or np.array): Ground truth labels (1 for clone, 0 for non-clone).
        threshold (float, optional): Distance cutoff. If None, finds the best threshold for F1.

    Returns:
        dict: A dictionary containing precision, recall, f1, accuracy, auc, and threshold.
    """
    distances = np.array(distances)
    labels = np.array(labels)

    # 1. Automatic Threshold Optimization (if threshold is not provided)
    if threshold is None:
        best_f1 = 0.0
        best_threshold = 0.0
        # Search across 100 potential thresholds between min and max distance
        threshold_candidates = np.linspace(distances.min(), distances.max(), 100)

        for t in threshold_candidates:
            # If distance < threshold, we predict it as a CLONE (1)
            preds = (distances < t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        threshold = best_threshold

    # 2. Final Prediction using the chosen threshold
    predictions = (distances < threshold).astype(int)

    # 3. Calculate Scores
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    accuracy = accuracy_score(labels, predictions)

    # 4. AUC calculation
    # Since smaller distance means higher probability of being a clone, we use negative distance
    try:
        auc = roc_auc_score(labels, -distances)
    except ValueError:
        auc = 0.0  # Handle cases with only one class in labels

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auc": auc,
        "threshold": threshold,
        "predictions" : predictions
    }


if __name__ == "__main__":
    # --- Local Functional Test ---
    # Mock data: 1 means Clone, 0 means Non-Clone
    test_labels = [1, 1, 0, 0, 0]

    # Mock distances: Lower distance for clones, higher for non-clones
    test_distances = [0.15, 0.32, 0.85, 1.10, 0.77]

    metrics = compute_clone_metrics(test_distances, test_labels)

    print(f"[*] Evaluation Results:")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1-Score:  {metrics['f1']:.4f}")
    print(f"  - AUC:       {metrics['auc']:.4f}")
    print(f"  - Optimized Threshold: {metrics['threshold']:.4f}")