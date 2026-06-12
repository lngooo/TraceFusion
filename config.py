import torch
import os


class Config:
    # This will be initialized in main.py
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Stage:
        TRAIN = 'train'
        VAL = 'val'
        TEST = 'test'

    # --- Path Management ---
    class Path:
        SOURCE_DIR_NATURAL = "./Dataset/LGL-DynT4/Data/Source_Clean_natural"
        SOURCE_DIR_OBF = "./Dataset/LGL-DynT4/Data/Source_Clean_obf"
        SOURCE_DIR = SOURCE_DIR_OBF
        TRACE_DIR = "./Dataset/LGL-DynT4/Data/Trace"
        TRAIN_CSV = "./Dataset/LGL-DynT4/Manifest/Train_Split.csv"
        VAL_CSV = "./Dataset/LGL-DynT4/Manifest/Val_Split.csv"
        TEST_CSV = "./Dataset/LGL-DynT4/Manifest/Test_Split.csv"
        CHECKPOINT_DIR = "./checkpointsNew"

    # --- Static Module (CodeBERT) ---
    class Static:
        # MODEL_NAME = 'microsoft/codebert-base'
        MODEL_NAME = 'microsoft/unixcoder-base-nine'
        OUTPUT_DIM = 768
        FREEZE_PARAMETERS = False # 冻结参数

    # --- Dynamic Module (BI-LSTM) ---
    class Dynamic:
        INPUT_DIM = 3
        HIDDEN_DIM = 512
        NUM_LAYERS = 2
        # MAX_SEQ_LEN = 512
        MAX_SEQ_LEN = 256
        OUTPUT_DIM = 768
        # 【新增】采样比例配置
        HEAD_RATIO = 0.5  # 保留前 50%
        TAIL_RATIO = 0.25  # 保留后 25%
        USE_RELATIVE_LOC = True  # 使用差分 LocID

    # --- Training ---
    class Training:
        # For single GPU, we set the actual Batch Size here
        TOTAL_BATCH_SIZE = 128
        ACCUMULATION_STEPS = 1
        LEARNING_RATE = 3e-5
        EPOCHS = 100
        MARGIN = 1.2
        DTW_GAMMA = 1.0 # Increase gamma for better stability
        NUM_WORKERS = 8  # Adjusted for single GPU throughput
        EMA_DECAY = 0.999
        EMA_START_EPOCH = 10
        MODALITY_DROPOUT_PROB = 0.2  # 每个模态被丢弃的概率
        LAMBDA_L1 = 0.001 # 0.001
        LAMBDA_L2 = 1  # 1.0

    class Ablation:
        USE_STATIC = True  # 是否使用代码文本模态
        USE_DYNAMIC = True  # 是否使用 Trace 模态

