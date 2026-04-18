#!/bin/bash
set -e

python -c "
import os
from huggingface_hub import snapshot_download


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# os.environ['HF_TOKEN'] = 'hf_individual_token'

model_dir = './gigachat_model'
print(f'donwloading model in  {model_dir}')

snapshot_download(
    repo_id='ai-sage/GigaChat3-10B-A1.8B-bf16',
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4
)
print('donwloaded')
"