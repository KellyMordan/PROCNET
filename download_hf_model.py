import os
from huggingface_hub import snapshot_download

model_name = 'hfl/chinese-roberta-wwm-ext'
model_dir = f'model_state/{model_name}'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

snapshot_download(repo_id=model_name, allow_patterns=['*.md', '*.txt', '*.json', '*.bin'], local_dir=model_dir, local_dir_use_symlinks=False)