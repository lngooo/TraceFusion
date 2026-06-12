import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.static_encoder import StaticEncoder
from Models.dynamic_encoder import DynamicEncoder
from Models.fusion_module import FusionModule
from config import Config


class UnifiedModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.static_encoder = StaticEncoder()
        self.dynamic_encoder = DynamicEncoder()
        self.fusion = FusionModule()

        self.use_static = Config.Ablation.USE_STATIC
        self.use_dynamic = Config.Ablation.USE_DYNAMIC

        print(f"[*] UnifiedModel Ablation | use_static={self.use_static}, use_dynamic={self.use_dynamic}")

        # --- 新增：给单模态消融单独的输出头 ---
        # 不影响 full model，只在 no_static / no_dynamic 时启用
        self.static_head = nn.Sequential(
            nn.LayerNorm(Config.Static.OUTPUT_DIM),
            nn.Linear(Config.Static.OUTPUT_DIM, 256)
        )

        self.dynamic_head = nn.Sequential(
            nn.LayerNorm(Config.Dynamic.OUTPUT_DIM),
            nn.Linear(Config.Dynamic.OUTPUT_DIM, 256)
        )

        # 冻结静态分支（复用 Config）
        if Config.Static.FREEZE_PARAMETERS:
            print("[*] Static Branch locked as feature extractor.")
            self.static_encoder.eval()

            for p in self.static_encoder.parameters():
                p.requires_grad = False
        else:
            print("[*] Static Branch: Partial Unlocking (Last 2 Layers).")
            for param in self.static_encoder.parameters():
                param.requires_grad = False

            unfreeze_layers = ['encoder.layer.10', 'encoder.layer.11', 'pooler']
            for name, param in self.static_encoder.named_parameters():
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True
                    print(f"[*] Unfreezing parameter: {name}")

            if hasattr(self.static_encoder.unixcoder.model, 'gradient_checkpointing_enable'):
                self.static_encoder.unixcoder.model.gradient_checkpointing_enable()
                print("[*] Static Branch: Gradient Checkpointing Enabled.")
            else:
                self.static_encoder.unixcoder.model.config.gradient_checkpointing = True
                print("[*] Static Branch: Gradient Checkpointing set via Config.")

    # -------------------------
    # Static Encoding
    # -------------------------
    def encode_static(self, input_ids, mask):
        return self.static_encoder(input_ids, mask)

    # -------------------------
    # Dynamic Encoding
    # -------------------------
    def encode_dynamic(self, trace, length):
        return self.dynamic_encoder(trace, lengths=length, return_sequence=True)

    # -------------------------
    # Helpers
    # -------------------------
    def pool_dynamic(self, seq_emb):
        """
        最小修改版本：先用简单 mean pooling。
        不改 full model 逻辑，只用于 no_static 的 dynamic-only ablation。
        """
        return seq_emb.mean(dim=1)

    # -------------------------
    # Fusion
    # -------------------------
    def fuse(self, s, d):
        s = F.normalize(s, p=2, dim=-1)
        d = F.normalize(d, p=2, dim=-1)

        h = self.fusion(s, d)

        return F.normalize(h, p=2, dim=-1)

    # -------------------------
    # Inference forward
    # -------------------------
    def forward(self, batch):
        # ========= 情况 1：full model，完全保持原逻辑 =========
        if self.use_static and self.use_dynamic:
            s1_emb = self.encode_static(batch['input_ids1'], batch['mask1'])
            s2_emb = self.encode_static(batch['input_ids2'], batch['mask2'])

            d11_seq_emb = self.encode_dynamic(batch['t11'], batch['t11_len'])
            d21_seq_emb = self.encode_dynamic(batch['t21'], batch['t21_len'])
            d12_seq_emb = self.encode_dynamic(batch['t12'], batch['t12_len']) if 't12' in batch else None
            d22_seq_emb = self.encode_dynamic(batch['t22'], batch['t22_len']) if 't22' in batch else None

            # --- 原来的 Modality Dropout 逻辑，保持不变 ---
            s1 = s1_emb
            s2 = s2_emb
            d11_seq = d11_seq_emb
            d21_seq = d21_seq_emb

            if self.training:
                prob = getattr(Config.Training, 'MODALITY_DROPOUT_PROB', 0.2)
                rand_val = torch.rand(1).item()

                if rand_val < prob:
                    s1 = torch.zeros_like(s1_emb)
                    s2 = torch.zeros_like(s2_emb)
                elif rand_val < (prob * 2):
                    d11_seq = torch.zeros_like(d11_seq_emb)
                    d21_seq = torch.zeros_like(d21_seq_emb)

            # --- 原来的监控逻辑，保持不变 ---
            with torch.no_grad():
                static_dist = torch.norm(
                    F.normalize(s1, p=2, dim=-1) - F.normalize(s2, p=2, dim=-1),
                    p=2, dim=1
                ).std().item()

                dynamic_dist = torch.norm(
                    d11_seq.mean(dim=1) - d21_seq.mean(dim=1),
                    p=2, dim=1
                ).std().item()

            # --- 原来的融合逻辑，保持不变 ---
            h1 = self.fuse(s1, d11_seq)
            h2 = self.fuse(s2, d21_seq)

            return {
                'h1': h1, 'h2': h2,
                'd11': d11_seq_emb, 'd12': d12_seq_emb,
                'd21': d21_seq_emb, 'd22': d22_seq_emb,
                'static_dist': static_dist,
                'dynamic_dist': dynamic_dist
            }

        # ========= 情况 2：static-only（no_dynamic） =========
        elif self.use_static and (not self.use_dynamic):
            s1_emb = self.encode_static(batch['input_ids1'], batch['mask1'])
            s2_emb = self.encode_static(batch['input_ids2'], batch['mask2'])

            h1 = F.normalize(self.static_head(s1_emb), p=2, dim=-1)
            h2 = F.normalize(self.static_head(s2_emb), p=2, dim=-1)

            with torch.no_grad():
                static_dist = torch.norm(h1 - h2, p=2, dim=1).std().item()
                dynamic_dist = 0.0

            return {
                'h1': h1, 'h2': h2,
                'd11': None, 'd12': None,
                'd21': None, 'd22': None,
                'static_dist': static_dist,
                'dynamic_dist': dynamic_dist
            }

        # ========= 情况 3：dynamic-only（no_static） =========
        elif (not self.use_static) and self.use_dynamic:
            d11_seq_emb = self.encode_dynamic(batch['t11'], batch['t11_len'])
            d21_seq_emb = self.encode_dynamic(batch['t21'], batch['t21_len'])
            d12_seq_emb = self.encode_dynamic(batch['t12'], batch['t12_len']) if 't12' in batch else None
            d22_seq_emb = self.encode_dynamic(batch['t22'], batch['t22_len']) if 't22' in batch else None

            d11_pool = self.pool_dynamic(d11_seq_emb)
            d21_pool = self.pool_dynamic(d21_seq_emb)

            h1 = F.normalize(self.dynamic_head(d11_pool), p=2, dim=-1)
            h2 = F.normalize(self.dynamic_head(d21_pool), p=2, dim=-1)

            with torch.no_grad():
                static_dist = 0.0
                dynamic_dist = torch.norm(h1 - h2, p=2, dim=1).std().item()

            return {
                'h1': h1, 'h2': h2,
                'd11': d11_seq_emb, 'd12': d12_seq_emb,
                'd21': d21_seq_emb, 'd22': d22_seq_emb,
                'static_dist': static_dist,
                'dynamic_dist': dynamic_dist
            }

        else:
            raise ValueError("Both static and dynamic branches are disabled.")