# main_mid_pos75_neg25.py
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import csv
import torch
import torch.optim as optim
import argparse

from tqdm import tqdm
import numpy as np
import random

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from Models.unified_model import UnifiedModel
from Modules.contrastive_loss import JointContrastiveLoss
from Modules.soft_dtw import SoftDTW

from Utils.data_loader import get_dataloader
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="HE-CodeRep Training with mid_pos75_neg25 Threshold Selection"
    )

    # 1. 基础环境
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # 2. 数据源切换 (Natural vs Obf)
    parser.add_argument('--data_mode', type=str, default='natural', choices=['natural', 'obf'])

    # 3. 模态开关 (消融实验核心)
    parser.add_argument('--no_static', action='store_true', help='Disable Static Branch')
    parser.add_argument('--no_dynamic', action='store_true', help='Disable Dynamic Branch')
    parser.add_argument('--freeze_static', action='store_true', help='FREEZE STATIC Branch')

    # 4. Loss 权重
    parser.add_argument('--l1', type=float, default=None)
    parser.add_argument('--l2', type=float, default=None)

    # 5. 其他
    parser.add_argument('--patience', type=int, default=10)

    return parser.parse_args()


def apply_config(args):
    if args.no_static and args.no_dynamic:
        raise ValueError("Cannot disable both static and dynamic branches at the same time.")

    Config.DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.data_mode == 'natural':
        Config.Path.SOURCE_DIR = Config.Path.SOURCE_DIR_NATURAL
    else:
        Config.Path.SOURCE_DIR = Config.Path.SOURCE_DIR_OBF

    Config.Ablation.USE_STATIC = not args.no_static
    Config.Ablation.USE_DYNAMIC = not args.no_dynamic
    Config.Static.FREEZE_PARAMETERS = args.freeze_static

    if args.l1 is not None:
        Config.Training.LAMBDA_L1 = args.l1
    if args.l2 is not None:
        Config.Training.LAMBDA_L2 = args.l2

    Config.Training.EARLY_STOPPING_PATIENCE = args.patience


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc_from_distance(scores, labels):
    """AUC where smaller distance means more likely clone."""
    try:
        from sklearn.metrics import roc_auc_score
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores, dtype=float)
        if len(np.unique(labels)) < 2:
            return float('nan')
        return float(roc_auc_score(labels, -scores))
    except Exception:
        return float('nan')


def binary_metrics_from_threshold(scores, labels, threshold):
    """Compute clone metrics using distance <= threshold as positive."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    preds = (scores <= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    balanced_acc = 0.5 * (recall + specificity)
    youden_j = recall + specificity - 1.0
    auc = safe_auc_from_distance(scores, labels)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'balanced_acc': float(balanced_acc),
        'youden_j': float(youden_j),
        'auc': float(auc),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'threshold': float(threshold),
    }


def compute_mid_pos75_neg25_threshold(scores, labels):
    """
    mid_pos75_neg25:
      threshold = (Q75(distance | positive clone) + Q25(distance | negative non-clone)) / 2

    由于 distance 越小越像 clone，这个阈值通常比 max-val-F1 更宽松，
    有助于提升 Natural setting 下的 recall 迁移。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos_dists = scores[labels == 1]
    neg_dists = scores[labels == 0]

    if len(pos_dists) == 0 or len(neg_dists) == 0:
        raise ValueError("Validation set must contain both positive and negative pairs.")

    pos_q75 = float(np.quantile(pos_dists, 0.75))
    neg_q25 = float(np.quantile(neg_dists, 0.25))
    threshold = (pos_q75 + neg_q25) / 2.0

    return threshold, pos_q75, neg_q25


def get_state_dict_to_save(model, ema_model, epoch, device):
    """
    与当前 validation 使用的模型保持一致：
    - EMA 启动前，保存原始 model；
    - EMA 启动后，保存 ema_model.module。
    """
    if epoch >= Config.Training.EMA_START_EPOCH:
        return {k: v.detach().cpu() for k, v in ema_model.module.state_dict().items()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def append_history_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    fieldnames = [
        'epoch', 'strategy', 'threshold', 'val_f1', 'val_precision', 'val_recall',
        'val_specificity', 'val_balanced_acc', 'val_youden_j', 'val_auc',
        'tp', 'fp', 'tn', 'fn', 'pos_mean', 'pos_std', 'pos_q75', 'neg_mean', 'neg_std', 'neg_q25'
    ]
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_some_epoch_model(model, ema_model, epoch, threshold, metrics, base_folder):
    current_epoch = epoch + 1
    if current_epoch == 1 or current_epoch % 10 == 0:
        save_path = os.path.join(base_folder, f"model_epoch_{current_epoch}.pth")
        torch.save({
            'model': get_state_dict_to_save(model, ema_model, epoch, Config.DEVICE),
            'epoch': current_epoch,
            'strategy': 'mid_pos75_neg25',
            'threshold': float(threshold),
            'f1': float(metrics['f1']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'auc': float(metrics['auc']),
        }, save_path)
        print(f"[*] Periodic checkpoint saved: model_epoch_{current_epoch}.pth")


# -------------------------
# validation
# -------------------------
def validate_mid_pos75_neg25(model, loader, device, epoch):
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        idx = 1
        for batch in tqdm(loader, desc="Validating", leave=False):
            if idx == 1:
                idx = 0
                print(f"[*] Raw Token IDs Check: {batch['input_ids1'][0][:20].tolist()}")

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

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels).astype(int)

    threshold, pos_q75, neg_q25 = compute_mid_pos75_neg25_threshold(scores_arr, labels_arr)
    metrics = binary_metrics_from_threshold(scores_arr, labels_arr, threshold)

    pos_dists = scores_arr[labels_arr == 1]
    neg_dists = scores_arr[labels_arr == 0]

    print("\n--- Distance Distribution on Val Set ---")
    print(f"Positive pairs (clone)   : mean = {pos_dists.mean():.4f}, std = {pos_dists.std():.4f}, q75 = {pos_q75:.4f}, count = {len(pos_dists)}")
    print(f"Negative pairs (non-clone): mean = {neg_dists.mean():.4f}, std = {neg_dists.std():.4f}, q25 = {neg_q25:.4f}, count = {len(neg_dists)}")
    print(f"[*] Strategy: mid_pos75_neg25 | Threshold = ({pos_q75:.6f} + {neg_q25:.6f}) / 2 = {threshold:.6f}")
    print(
        f"[*] Validation Epoch {epoch + 1} | "
        f"F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | "
        f"AUC: {metrics['auc']:.4f}"
    )

    dist_stats = {
        'pos_mean': float(pos_dists.mean()),
        'pos_std': float(pos_dists.std()),
        'pos_q75': float(pos_q75),
        'neg_mean': float(neg_dists.mean()),
        'neg_std': float(neg_dists.std()),
        'neg_q25': float(neg_q25),
    }

    return metrics, threshold, dist_stats


# -------------------------
# train
# -------------------------
def train():
    device = Config.DEVICE
    print("[*] Initializing Training Environment...")

    model = UnifiedModel().to(device)

    ema_decay = getattr(Config.Training, "EMA_DECAY", 0.999)
    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
    ).cpu()

    dtw_crit = SoftDTW(gamma=Config.Training.DTW_GAMMA).to(device)
    cont_crit = JointContrastiveLoss(margin=Config.Training.MARGIN).to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=Config.Training.LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.Training.EPOCHS
    )

    train_loader = get_dataloader(
        Config.Path.TRAIN_CSV,
        Config.Training.TOTAL_BATCH_SIZE,
        Config.Path.SOURCE_DIR,
        Config.Path.TRACE_DIR,
        Config.Training.NUM_WORKERS,
        mode=Config.Stage.TRAIN
    )

    val_loader = get_dataloader(
        Config.Path.VAL_CSV,
        Config.Training.TOTAL_BATCH_SIZE,
        Config.Path.SOURCE_DIR,
        Config.Path.TRACE_DIR,
        Config.Training.NUM_WORKERS,
        mode=Config.Stage.VAL
    )

    base_folder = os.path.join(Config.Path.CHECKPOINT_DIR, "gpu" + str(Config.DEVICE))
    os.makedirs(base_folder, exist_ok=True)
    history_csv = os.path.join(base_folder, "mid_pos75_neg25_history.csv")

    best_f1 = 0.0
    best_balanced_acc = 0.0

    print(f"[*] Training started on {device}")
    print(f"[*] SEED CHECK - Random Value: {torch.randn(1).item():.6f}")

    acc_steps = Config.Training.ACCUMULATION_STEPS if hasattr(Config.Training, 'ACCUMULATION_STEPS') else 1

    for epoch in range(Config.Training.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 0:
                example_code = train_loader.dataset.tokenizer.decode(batch['input_ids1'][0])
                print(f"DEBUG - Current Code Sample (First 50 chars): {example_code[:50]}")

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out = model(batch)
            labels = batch['label']

            if Config.Ablation.USE_DYNAMIC:
                t11_len, t21_len = batch['t11_len'], batch['t21_len']
                len_ab = (t11_len + t21_len) / 2 + 1e-5
                dtw_ab = dtw_crit(out['d11'], out['d21']) / len_ab
                loss_l1_pos = dtw_ab[labels == 1].mean() if (labels == 1).any() else torch.tensor(0.0, device=device)
                loss_l1_neg = torch.relu(0.2 - dtw_ab[labels == 0]).mean() if (labels == 0).any() else torch.tensor(0.0, device=device)
                loss_l1 = loss_l1_pos + loss_l1_neg
            else:
                loss_l1 = torch.tensor(0.0, device=device)

            loss_l2 = cont_crit(out['h1'], out['h2'], labels).mean()
            total_loss = (Config.Training.LAMBDA_L2 * loss_l2 + Config.Training.LAMBDA_L1 * loss_l1)

            if torch.isnan(total_loss):
                continue

            total_loss = total_loss / acc_steps
            total_loss.backward()

            if (batch_idx + 1) % acc_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if epoch >= Config.Training.EMA_START_EPOCH:
                    is_first_step = (batch_idx + 1) <= acc_steps
                    if epoch == Config.Training.EMA_START_EPOCH and is_first_step:
                        with torch.no_grad():
                            for ema_p, model_p in zip(ema_model.module.parameters(), model.parameters()):
                                ema_p.copy_(model_p)
                        print(f"\n[*] EMA Model initialized at Epoch {epoch + 1}, Batch {batch_idx + 1}")
                    ema_model.update_parameters(model)

            pbar.set_postfix({
                "L1": f"{loss_l1.item():.3f}",
                "L2": f"{loss_l2.item():.3f}",
                "total": f"{total_loss.item():.3f}",
            })

        scheduler.step()

        if epoch >= Config.Training.EMA_START_EPOCH:
            val_target = ema_model.module.to(device)
            metrics, threshold, dist_stats = validate_mid_pos75_neg25(val_target, val_loader, device, epoch)
            val_target.cpu()
        else:
            metrics, threshold, dist_stats = validate_mid_pos75_neg25(model, val_loader, device, epoch)

        param_sum = sum(p.sum().item() for p in model.parameters() if p.requires_grad)
        print(f"[*] Model Checksum: {param_sum:.10f}")

        append_history_csv(history_csv, {
            'epoch': epoch + 1,
            'strategy': 'mid_pos75_neg25',
            'threshold': threshold,
            'val_f1': metrics['f1'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'val_specificity': metrics['specificity'],
            'val_balanced_acc': metrics['balanced_acc'],
            'val_youden_j': metrics['youden_j'],
            'val_auc': metrics['auc'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn'],
            **dist_stats,
        })

        save_some_epoch_model(model, ema_model, epoch, threshold, metrics, base_folder)

        # 主要保存准则：mid_pos75_neg25 下的 validation F1 最大
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_path_named = os.path.join(base_folder, "best_model_mid_pos75_neg25.pth")
            save_path_compat = os.path.join(base_folder, "best_model.pth")
            payload = {
                'model': get_state_dict_to_save(model, ema_model, epoch, device),
                'epoch': epoch + 1,
                'strategy': 'mid_pos75_neg25',
                'threshold': float(threshold),
                'f1': float(metrics['f1']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'auc': float(metrics['auc']),
                'balanced_acc': float(metrics['balanced_acc']),
                'youden_j': float(metrics['youden_j']),
            }
            torch.save(payload, save_path_named)
            torch.save(payload, save_path_compat)
            print(
                f"[!] New Best mid_pos75_neg25 F1: {best_f1:.4f}, "
                f"TH={threshold:.6f}, saved=best_model_mid_pos75_neg25.pth and best_model.pth"
            )

        # 额外保存 balanced accuracy 最优版本，便于后续诊断；正式结果仍建议用 F1 最优版
        if metrics['balanced_acc'] > best_balanced_acc:
            best_balanced_acc = metrics['balanced_acc']
            save_path_bal = os.path.join(base_folder, "best_model_mid_pos75_neg25_balacc.pth")
            torch.save({
                'model': get_state_dict_to_save(model, ema_model, epoch, device),
                'epoch': epoch + 1,
                'strategy': 'mid_pos75_neg25',
                'selection_metric': 'balanced_acc',
                'threshold': float(threshold),
                'f1': float(metrics['f1']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'auc': float(metrics['auc']),
                'balanced_acc': float(metrics['balanced_acc']),
                'youden_j': float(metrics['youden_j']),
            }, save_path_bal)
            print(f"[!] New Best mid_pos75_neg25 BalAcc: {best_balanced_acc:.4f}, saved=best_model_mid_pos75_neg25_balacc.pth")


def print_config(args):
    print("=" * 30 + " CONFIGURATION SUMMARY " + "=" * 30)
    print(f"[*] Device:        {Config.DEVICE}")
    print(f"[*] Source Dir:    {Config.Path.SOURCE_DIR}")
    print(f"[*] Seed:          {args.seed}")
    print(f"[*] Threshold Strategy: mid_pos75_neg25 = (Val positive Q75 + Val negative Q25) / 2")
    print(f"[*] Ablation Settings:")
    print(f"    - Lambda (L1/L2):  {Config.Training.LAMBDA_L1} / {Config.Training.LAMBDA_L2}")
    print(f"    - Use Static/Dynamic: {Config.Ablation.USE_STATIC} / {Config.Ablation.USE_DYNAMIC}")
    print(f"    - Freeze Static:      {Config.Static.FREEZE_PARAMETERS}")
    print(f"[*] Training Hyperparameters:")
    print(f"    - Learning Rate:      {Config.Training.LEARNING_RATE}")
    print(f"    - Batch Size:         {Config.Training.TOTAL_BATCH_SIZE}")
    print(f"    - Early Stop Patience: {Config.Training.EARLY_STOPPING_PATIENCE}")
    print("=" * 83)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    apply_config(args)
    print_config(args)
    train()
