import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class SoftDTW(nn.Module):
    def __init__(self, gamma=Config.Training.DTW_GAMMA):
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.K = 3  # number of predecessors

    def forward(self, x, y):
        """
        Diagonal vectorized Soft-DTW.
        x: [B, N, D], y: [B, M, D]
        """
        B, N, D = x.shape
        M = y.shape[1]
        device = x.device

        # 1. Compute pairwise distances
        dist_mat = torch.cdist(x, y, p=2)

        # 2. Initialize DP table with a safe large value
        # Using 1e12 to prevent exp() overflow while keeping it "infinite"
        v = torch.full((B, N + 1, M + 1), 1e12, device=device)
        v[:, 0, 0] = 0

        # 3. Diagonal-based vectorized update
        for k in range(1, N + M):
            i_min = max(1, k - M + 1)
            i_max = min(k, N)
            i_idx = torch.arange(i_min, i_max + 1, device=device)
            j_idx = k - i_idx + 1

            # Predecessors: [B, diag_len, 3]
            r0 = v[:, i_idx - 1, j_idx - 1]
            r1 = v[:, i_idx - 1, j_idx]
            r2 = v[:, i_idx, j_idx - 1]

            combined = torch.stack([r0, r1, r2], dim=-1)

            # Soft-min logic: -gamma * logsumexp(-combined / gamma)
            # We add a small epsilon inside exp if needed, but logsumexp is stable
            soft_min = -self.gamma * torch.logsumexp(-combined / self.gamma, dim=-1) + self.gamma * math.log(self.K)

            v[:, i_idx, j_idx] = dist_mat[:, i_idx - 1, j_idx - 1] + soft_min

        return v[:, N, M]


# --- Test Main Function ---
if __name__ == "__main__":
    # Simulate your training environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtw = SoftDTW(gamma=1.0).to(device)

    # B=64, Seq=128, Dim=512
    # IMPORTANT: We use F.normalize to mimic the fix for negative loss
    x = F.normalize(torch.randn(64, 128, 512, device=device), p=2, dim=-1)
    y = F.normalize(torch.randn(64, 128, 512, device=device), p=2, dim=-1)

    print(f"[*] Testing Soft-DTW on {device}...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    loss = dtw(x, y).mean()
    end.record()

    torch.cuda.synchronize()
    print(f"[+] Loss Value: {loss.item():.4f}")
    print(f"[+] Execution Time: {start.elapsed_time(end):.2f} ms")

    if loss < 0:
        print("[!] Warning: Loss is still negative. Normalization is required in main.py!")
    else:
        print("[OK] Loss is positive. Stabilization successful.")