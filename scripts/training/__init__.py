# zgx conda env has the compatible GPU fine-tuning stack (unsloth, trl, torch+cu130).
# TRITON_PTXAS_BLACKWELL_PATH points Triton to the conda env's ptxas (CUDA 13.2)
# which supports GB10 (sm_121) natively, replacing the old TRITON_INTERPRET=1 workaround.
FINETUNE_PYTHON = '/home/zgx/miniforge3/envs/zgx/bin/python'
FINETUNE_PTXAS  = '/home/zgx/miniforge3/envs/zgx/bin/ptxas'
