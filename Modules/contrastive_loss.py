import torch
import torch.nn as nn
import torch.nn.functional as F


class JointContrastiveLoss(nn.Module):
    """
    Implementation of L3 loss for the final joint representations (HA, HB).
    Uses the clone label L to pull similar functions together or push others apart.
    """

    def __init__(self, margin=1.0):
        super(JointContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, h_a, h_b, label):
        # Calculate Euclidean distance d
        dist = F.pairwise_distance(h_a, h_b, p=2)

        # Pull: L * 0.5 * d^2
        loss_pull = label * 0.5 * torch.pow(dist, 2)

        # Push: (1-L) * 0.5 * max(0, m-d)^2
        loss_push = (1 - label) * 0.5 * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)

        return torch.mean(loss_pull + loss_push)