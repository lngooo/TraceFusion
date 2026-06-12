import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from Models.unixcoder import UniXcoder
from config import Config


class StaticEncoder(nn.Module):
    """
    Static Perception Branch: Extracts semantic features from C source code.
    Now uses UniXcoder as the backbone (replaced CodeBERT) for better AST-aware representation.
    """

    def __init__(self,
                 model_name=getattr(Config.Static, 'MODEL_NAME', 'microsoft/unixcoder-base-nine'),
                 output_dim=getattr(Config.Static, 'OUTPUT_DIM', 768),
                 use_normalization=True):
        super(StaticEncoder, self).__init__()

        # Initialize UniXcoder backbone via its official wrapper
        self.unixcoder = UniXcoder(model_name)
        self.device = Config.DEVICE
        self.unixcoder.to(self.device)

        # Projection layer to align static features with dynamic features
        self.fc = nn.Linear(768, output_dim)

        self.use_norm = use_normalization
        if self.use_norm:
            self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, input_ids, attention_mask):
        # """
        # Forward pass using CLS token representation from UniXcoder.
        # Args:
        #     input_ids: Tokenized indices [batch, seq_len]
        #     attention_mask: Mask for padding [batch, seq_len]
        # """
        # # UniXcoder's internal model (self.unixcoder.model) is a standard HF model
        # # It accepts input_ids and attention_mask like CodeBERT.
        # outputs = self.unixcoder.model(
        #     input_ids=input_ids.to(self.device),
        #     attention_mask=attention_mask.to(self.device),
        #     return_dict=True
        # )
        #
        # # CLS token is at index 0
        # cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        #
        # latent_vec = self.fc(cls_embeddings)
        _, sentence_embeddings = self.unixcoder(input_ids.to(self.device))
        latent_vec = self.fc(sentence_embeddings)

        if self.use_norm:
            latent_vec = self.layer_norm(latent_vec)

        return latent_vec


# --- Main Test Block (identical to original, only model_name updated) ---
if __name__ == "__main__":
    # 1. Configuration for testing
    device = Config.DEVICE
    test_file = "../Dataset/LGL-DynT4/Data/Source_Clean_obf/F01_Sum/A01_F01_S_Loop.c"
    model_name = getattr(Config.Static, 'MODEL_NAME', 'mmicrosoft/unixcoder-base-nine')  # Updated default

    # 2. Initialize Model and Tokenizer
    print(f"[*] Loading model and tokenizer: {model_name}")
    model = StaticEncoder(model_name=model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Perform test on the specific .c file
    if os.path.exists(test_file):
        try:
            print(f"[*] Reading test file: {test_file}")
            with open(test_file, 'r', encoding='utf-8') as f:
                code_content = f.read()

            # Preprocess code string into tensors
            inputs = tokenizer(
                code_content,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=512
            ).to(device)

            # Forward pass
            model.eval()
            with torch.no_grad():
                embedding = model(inputs['input_ids'], inputs['attention_mask'])

            print(f"[Success] Successfully encoded .c file.")
            print(f"-> Embedding Shape: {embedding.shape}")
            print(f"-> Embedding Sample (first 5 values): {embedding[0, :5].cpu().numpy()}")

        except Exception as e:
            print(f"[Error] Test failed during execution: {e}")
    else:
        print(f"[Skip] Test file not found at: {test_file}")