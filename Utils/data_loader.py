import torch
import os
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from Utils.trace_processor import TraceProcessor
from config import Config
import re


class CodeTraceDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, source_base, trace_base, mode=Config.Stage.TRAIN):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # 从配置中获取 MAX_SEQ_LEN
        self.processor = TraceProcessor(max_seq_len=Config.Dynamic.MAX_SEQ_LEN)
        self.source_base = source_base
        self.trace_base = trace_base
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _get_trace_path(self, algo_rel_path, t_index):
        folder_path = os.path.join(self.trace_base, algo_rel_path)
        prefix = f"T{t_index:02d}"
        if not os.path.exists(folder_path): return None
        for f in os.listdir(folder_path):
            if f.startswith(prefix) and f.endswith(".csv"):
                return os.path.join(folder_path, f)
        return None

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 适配你的 Manifest 列名：algorithm1, algorithm2
        a1, a2 = row['algorithm1'], row['algorithm2']

        # 假设 Source 文件名和 Trace 文件夹结构一致，可以从 a1 中推导
        # 如果 source 路径结构不同，请根据实际手动调整拼接逻辑
        c1_path = os.path.join(self.source_base, a1 + ".c")
        c2_path = os.path.join(self.source_base, a2 + ".c")

        # 读取源码 (带异常处理防止文件缺失)
        try:
            with open(c1_path, 'r', encoding='utf-8') as f:
                c1 = f.read()
            with open(c2_path, 'r', encoding='utf-8') as f:
                c2 = f.read()
        except:
            c1, c2 = "", ""

        # 采样逻辑
        if self.mode in [Config.Stage.VAL, Config.Stage.TEST]:
            idx_shared, idx_alt = 9, 10
        else:
            idx_shared = random.randint(1, 10)
            idx_alt = random.choice([i for i in range(1, 11) if i != idx_shared])

        # 获取 Trace 特征
        t11_feat, t11_len = self.processor.process_csv(self._get_trace_path(a1, idx_shared))
        t12_feat, t12_len = self.processor.process_csv(self._get_trace_path(a1, idx_alt))
        t21_feat, t21_len = self.processor.process_csv(self._get_trace_path(a2, idx_shared))
        t22_feat, t22_len = self.processor.process_csv(self._get_trace_path(a2, idx_alt))

        data = {
            't11': t11_feat, 't11_len': t11_len,
            't12': t12_feat, 't12_len': t12_len,
            't21': t21_feat, 't21_len': t21_len,
            't22': t22_feat, 't22_len': t22_len,
            'label': torch.tensor(row['label'], dtype=torch.float),
            'algorithm1': re.search(r'(A\d+_F\d+)', a1).group(1),
            'algorithm2': re.search(r'(A\d+_F\d+)', a2).group(1),
        }

        # CodeBERT Tokenization
        t1 = self.tokenizer(c1, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        t2 = self.tokenizer(c2, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        data.update({
            'input_ids1': t1['input_ids'].squeeze(0), 'mask1': t1['attention_mask'].squeeze(0),
            'input_ids2': t2['input_ids'].squeeze(0), 'mask2': t2['attention_mask'].squeeze(0)
        })
        return data


def get_dataloader(csv_path, batch_size, source_base, trace_base, num_workers=4, mode=Config.Stage.TRAIN):
    ds = CodeTraceDataset(csv_path, Config.Static.MODEL_NAME, source_base, trace_base, mode=mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == Config.Stage.TRAIN), num_workers=num_workers,
                      pin_memory=True)