import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import Config

class DynamicEncoder(nn.Module):
    def __init__(self,
                 input_dim=Config.Dynamic.INPUT_DIM,
                 hidden_dim=512,
                 num_layers=2,
                 output_dim=768):
        super(DynamicEncoder, self).__init__()

        # 1. 映射层：处理 (Value, LocID, OpType) 三维输入
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 2. 核心双向 LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths=None, return_sequence=False):
        """
        x: [batch, seq_len, input_dim]
        lengths: [batch] 存储每个样本的真实长度
        """
        # 投影特征
        x = self.input_projection(x) # [batch, seq_len, 64]

        if lengths is not None:
            # 确保 lengths 在 CPU 上（pack_padded_sequence 的要求）
            lengths_cpu = lengths.cpu()
            # 压紧序列，忽略 Padding 部分
            packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, (hidden, _) = self.lstm(packed_x)
            # 解开压紧的序列
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            lstm_out, (hidden, _) = self.lstm(x)

        lstm_out = self.norm(lstm_out)

        # 如果用于对齐损失 (Soft-DTW)，返回完整序列
        if return_sequence:
            return self.fc(lstm_out) # [batch, seq_len, output_dim]

        # 否则返回最后一个有效时间步的状态（用于特征融合）
        # 取双向 LSTM 的最后一层隐藏状态并拼接
        # hidden 形状: [num_layers * num_directions, batch, hidden_dim]
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.norm(out))