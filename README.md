# HE-CodeRep

## Installation Guide

This guide describes how to set up the environment on a server with **NVIDIA Tesla V100S (32GB)** GPUs.

### 1. System Environment
- **GPU**: 4 × NVIDIA Tesla V100S-PCIE-32GB  
- **Driver Version**: 550.120  
- **Maximum CUDA Support**: 12.4  
- **OS**: Linux (recommended)

### 2. Creating a Conda Environment
We use Python 3.12 for optimal compatibility with modern Transformer libraries.

```bash
# Create and activate the environment
conda create -n lgl_hecoderep python=3.12 -y
conda activate lgl_hecoderep

# Install PyTorch with CUDA 12.1 support via PyPI (ensures compatible internal libraries)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# Verify the installation
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```
### 3. Installing Additional Dependencies
Install the remaining Python packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```

