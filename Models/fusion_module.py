import torch
import torch.nn as nn
from config import Config


class CrossModalAttention(nn.Module):
    """
    Implements the Cross-Attention mechanism seen in Section 5 of the architecture.
    It allows the Static features to attend to Dynamic features and vice versa.
    """

    def __init__(self, embed_dim=768, num_heads=8):
        super(CrossModalAttention, self).__init__()
        # MultiheadAttention handles the core cross-attention logic
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        """
        Args:
            query: Anchor modality (e.g., Static Embedding)
            key/value: Context modality (e.g., Dynamic Sequence)
        """
        # attn_output: the enhanced representation of the query based on key/value context
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + attn_output)  # Residual connection


class FusionModule(nn.Module):
    """
    The updated Fusion Module using Cross-Attention as per the architecture diagram.
    """

    def __init__(self):
        super(FusionModule, self).__init__()
        # In the diagram, static and dynamic features meet at the Cross-Attention node
        self.cross_attn = CrossModalAttention(embed_dim=768)

        # Final layers to produce the 'Joint Representation'
        self.fc_joint = nn.Linear(768, 256)

    def forward(self, static_emb, dynamic_seq):
        """
        Args:
            static_emb: [batch, 1, 768] (Static branch output)
            dynamic_seq: [batch, seq_len, 768] (Dynamic branch hidden states)
        """
        # Step 5: Static features attend to Dynamic hidden states
        # static_emb acts as Query; dynamic_seq acts as Key and Value
        if static_emb.dim() == 2:
            static_emb = static_emb.unsqueeze(1)

        fused_feat = self.cross_attn(static_emb, dynamic_seq, dynamic_seq)

        # Squeeze the sequence dimension for the joint vector
        joint_rep = self.fc_joint(fused_feat.squeeze(1))
        return joint_rep