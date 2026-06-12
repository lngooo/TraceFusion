import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'   # 只使用 GPU 3
import torch
from tqdm import tqdm
from Models.static_encoder import StaticEncoder
from Models.dynamic_encoder import DynamicEncoder
from Models.fusion_module import FusionModule
from Utils.data_loader import get_dataloader
from Utils.metrics import compute_clone_metrics
from config import Config
import torch.nn.functional as F
import pandas as pd
from Models.unified_model import UnifiedModel

def test():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"HEAD_RATIO: {Config.Dynamic.HEAD_RATIO}")
    print(f"TAIL_RATIO: {Config.Dynamic.TAIL_RATIO}")
    print(f"USE_RELATIVE_LOC: {Config.Dynamic.USE_RELATIVE_LOC}")
    print(f"MAX_SEQ_LEN: {Config.Dynamic.MAX_SEQ_LEN}")

    # --- 1. Infrastructure Setup ---
    # Use the best available GPU found during training config
    # device = Config.get_best_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Testing on device: {device}")

    # --- 2. Initialize Models ---
    model = UnifiedModel().to(device)

    DATA_MODE = "natural"  # "natural" or "obf"

    # --- 3. Load Trained Weights ---
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "p10_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "p1_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "p0.1_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "p0.01_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "p0.001_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "p0.0001_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "p0_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nostatic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nodynamic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "freezestatic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_p0.001_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "seed123", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "seed0", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "seed2026", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "seed666", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_nostatic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_nodynamic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_freezestatic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.0001_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.1_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p1_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p10_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_0.01_nostatic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_0.01_nodynamic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_0.01_freezestatic", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_0.01_seed0", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_0.01_seed123", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_0.01_seed666", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_0.01_seed2026", "best_model.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_10.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_20.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_30.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_40.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_50.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_60.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_70.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_80.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_90.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "natural_p0.01_1", "model_epoch_100.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_max_val_balacc.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_max_val_f1.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_max_val_youden.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_mid_pos75_neg25.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_mid_pos90_neg10.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_p95_max_recall.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_val_pos_q90.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0", "best_model_val_pos_q95.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_max_val_balacc.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_max_val_f1.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_max_val_youden.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_mid_pos75_neg25.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_mid_pos90_neg10.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_p95_max_recall.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_val_pos_q90.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.01", "best_model_val_pos_q95.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_max_val_balacc.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_max_val_f1.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_max_val_youden.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_mid_pos75_neg25.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_mid_pos90_neg10.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_p95_max_recall.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_val_pos_q90.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_natural_0.001", "best_model_val_pos_q95.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_max_val_balacc.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_max_val_f1.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_max_val_youden.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_mid_pos75_neg25.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_mid_pos90_neg10.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_p95_max_recall.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_val_pos_q90.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.001", "best_model_val_pos_q95.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_max_val_balacc.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_max_val_f1.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_max_val_youden.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_mid_pos75_neg25.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_mid_pos90_neg10.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_p95_max_recall.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_val_pos_q90.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "stra_obf_0.01", "best_model_val_pos_q95.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0.0001", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0.001", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0.01", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0.1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_1", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_10", "best_model.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0_seed0", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0_seed123", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0_seed666", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0_seed2026", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0.01_seed123", "best_model.pth")
    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0.01_seed666", "best_model.pth")

    # checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0_nostatic", "best_model.pth")
    checkpoint_path = os.path.join(Config.Path.CHECKPOINT_DIR, "nat_0_nodynamic", "best_model.pth")

    if DATA_MODE == "natural":
        Config.Path.SOURCE_DIR = Config.Path.SOURCE_DIR_NATURAL
    elif DATA_MODE == "obf":
        Config.Path.SOURCE_DIR = Config.Path.SOURCE_DIR_OBF

    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Checkpoint not found at {checkpoint_path}. Did you finish training?")
        return

    print(f"[*] Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dicts (using the keys defined in your main.py saving logic)
    model.load_state_dict(checkpoint["model"])
    best_threshold = checkpoint.get('threshold', None)
    print(f"[*] Successfully loaded model from Epoch {checkpoint['epoch']} (Best F1={checkpoint['f1']:.4f}, Best Threshold= {best_threshold:.4f})")

    # --- 4. Data Preparation ---
    test_loader = get_dataloader(
        Config.Path.TEST_CSV,
        Config.Training.TOTAL_BATCH_SIZE,
        Config.Path.SOURCE_DIR,
        Config.Path.TRACE_DIR,
        Config.Training.NUM_WORKERS,
        mode=Config.Stage.TEST
    )

    print("### SOURCE_DIR:", Config.Path.SOURCE_DIR)

    # --- 5. Evaluation Loop ---
    model.eval()

    scores, labels,algorithm1,algorithm2 = [], [], [], []

    print(f"[*] Starting Evaluation on Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            out = model({
                'input_ids1': batch['input_ids1'].to(device),
                'mask1': batch['mask1'].to(device),
                't11': batch['t11'].to(device),
                't11_len': batch['t11_len'].to(device),
                'input_ids2': batch['input_ids2'].to(device),
                'mask2': batch['mask2'].to(device),
                't21': batch['t21'].to(device),
                't21_len': batch['t21_len'].to(device)
            })
            distance = torch.norm(out['h1'] - out['h2'], p=2, dim=1)

            scores.extend(distance.cpu().numpy())
            labels.extend(batch['label'].cpu().numpy())
            if 'algorithm1' in batch:
                algorithm1.extend(batch['algorithm1'])
                algorithm2.extend(batch['algorithm2'])

    # --- 6. Results Calculation ---
    metrics = compute_clone_metrics(scores, labels, threshold=best_threshold)

    #############3suc
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    pos_dists = scores_arr[labels_arr == 1]
    neg_dists = scores_arr[labels_arr == 0]

    print("\n--- Distance Distribution on Test Set ---")
    print(
        f"Positive pairs (clone)   : mean = {pos_dists.mean():.4f}, std = {pos_dists.std():.4f}, count = {len(pos_dists)}")
    print(
        f"Negative pairs (non-clone): mean = {neg_dists.mean():.4f}, std = {neg_dists.std():.4f}, count = {len(neg_dists)}")
    #############3suc

    print("\n" + "=" * 30)
    print("      TEST SET RESULTS")
    print("=" * 30)
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC Score: {metrics['auc']:.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print("=" * 30)

    ### 4 保存下预测数据
    predictions = metrics['predictions']
    assert len(algorithm1) == len(algorithm2) == len(labels) == len(predictions)
    df = pd.DataFrame({
        'algorithm1': algorithm1,
        'algorithm2': algorithm2,
        'scores': scores,
        'label': labels,
        'predictions': predictions
    })
    # 强制 label 列为整数类型
    df['label'] = df['label'].astype(int)
    df.to_csv('out/test_predictions.csv', index=False)
    print("[*] Predictions saved to test_predictions.csv")
    ### 4 保存下预测数据

if __name__ == "__main__":
    test()