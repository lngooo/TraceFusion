import pandas as pd
import numpy as np
import torch
import os
import hashlib
from config import Config

class TraceProcessor:
    def __init__(self, max_seq_len=Config.Dynamic.MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len

    def parse_value(self, v):
        """
                核心处理逻辑：
                1. 尝试直接转浮点数
                2. 尝试转16进制（如内存地址）
                3. 失败则进行 MD5 哈希，映射到固定区间，保留字符语义
                4. 最后进行 log1p 缩放
                """
        s = str(v).strip()
        try:
            # 尝试直接解析为数字 (包括 10.5, 100 等)
            val = float(s)
        except ValueError:
            try:
                # 尝试解析为 16 进制 (包括 0xABC, 4005d1 等)
                val = float(int(s, 16))
            except ValueError:
                # 兜底：处理特殊字符如 '#', '￥@', 'INPUT'
                # 使用 MD5 确保相同的字符映射到相同的数值
                val = float(int(hashlib.md5(s.encode()).hexdigest(), 16) % 10000)

        # Log 归一化：压制可能的大数（尤其是内存地址），保留符号
        return np.sign(val) * np.log1p(np.abs(val))

    def process_csv(self, file_path):
        empty_feat = torch.zeros((self.max_seq_len, 3), dtype=torch.float32)
        empty_len = torch.tensor(0, dtype=torch.long)

        if file_path is None or not os.path.exists(file_path):
            return empty_feat, empty_len

        try:
            df = pd.read_csv(file_path, skiprows=1, header=0)
            if df.empty: return empty_feat, empty_len

            # 1. 提取所有操作并保持时间顺序
            # 标记 OpType: INPUT=0, S=1
            df['op_type'] = df['op'].apply(lambda x: 0 if x == 'INPUT' else 1)

            # 2. 处理 LocID 特征：计算差分 (Stride)
            # 既然原始 loc_id 是递增的，差分能体现出访问的“步长”
            # 比如顺序存储差分为1，跳跃存储差分>1
            loc_array = df['loc_id'].values.astype(float)
            loc_diff = np.zeros_like(loc_array)
            if Config.Dynamic.USE_RELATIVE_LOC:
                loc_diff[1:] = loc_array[1:] - loc_array[:-1]
            else:
                loc_diff[1:] = loc_array[1:]

            # 3. 构造特征矩阵 (Value, LocDiff, OpType)
            values = df['value'].apply(self.parse_value).values
            op_types = df['op_type'].values

            raw_data = np.stack([values, loc_diff, op_types], axis=1).astype(np.float32)
            curr_len = len(raw_data)

            # 4. 自适应采样 (针对 >512 的长文件)
            if curr_len > self.max_seq_len:
                # 保留前 256 个点（核心逻辑）和最后 128 个点（结果），中间等间距采样
                head_size = int(self.max_seq_len * Config.Dynamic.HEAD_RATIO)
                tail_size = int(self.max_seq_len * Config.Dynamic.TAIL_RATIO)
                mid_size = self.max_seq_len - head_size - tail_size

                head = raw_data[:head_size]
                tail = raw_data[-tail_size:]
                mid_pool = raw_data[head_size:-tail_size]

                mid_indices = np.linspace(0, len(mid_pool) - 1, mid_size).astype(int)
                mid = mid_pool[mid_indices]

                final_data = np.vstack([head, mid, tail])
                final_len = self.max_seq_len
            else:
                final_data = raw_data
                final_len = curr_len

            # 5. 组装并 Padding
            feature_matrix = np.zeros((self.max_seq_len, 3), dtype=np.float32)
            feature_matrix[:final_len, :] = final_data

            return torch.from_numpy(feature_matrix), torch.tensor(final_len, dtype=torch.long)

        except Exception as e:
            # 加上这一行，运行训练时如果报错，就能看到是哪个文件出了问题
            print(f"[Warning] Error processing file: {file_path}. Error: {e}")
            return empty_feat, empty_len


# --- 下面是你要的测试函数 ---
if __name__ == "__main__":
    # 配置测试路径
    # test_path = "../Dataset/LGL-DynT4/Data/Trace/F02_GCD/A02_F02_O_A01_Flat/T01_A02_F02.csv"
    # test_path = "../Dataset/LGL-DynT4/Data/Trace/F03_Factorial/A05_F03_O_A04_Opaque/T08_A05_F03.csv"
    test_path = "../Dataset/LGL-DynT4/Data/Trace/F47_XorCipher/A01_F47_S_BasicIter/T04_A01_F47.csv"

    # 实例化处理器（假设 max_seq_len 为 256）
    processor = TraceProcessor(max_seq_len=Config.Dynamic.MAX_SEQ_LEN)

    feat, length = processor.process_csv(test_path)

    print("-" * 50)
    if length > 0:
        print(f"[Success] 处理成功！")
        print(f"有效序列长度 (Length): {length.item()}")
        print(f"特征矩阵形状 (Shape): {feat.shape}")  # 应该是 [256, 3]

        # 打印前 5 个时间步的详细数据
        print("\n前 30 个时间步的数据 [Value, LocID, OpType]:")
        # .numpy() 方便打印
        print(feat[:30].numpy())

        # 统计分布
        op_types = feat[:length, 2].numpy()
        input_count = np.sum(op_types == 0)
        store_count = np.sum(op_types == 1)
        print(f"\n分布统计: INPUT 记录数 = {input_count}, STORE 记录数 = {store_count}")
    else:
        print("[Fail] 未能读取到有效数据，请检查文件路径或文件内容格式。")
    print("-" * 50)