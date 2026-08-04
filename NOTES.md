# How to install everything

```bash
git clone https://github.com/evaportelance/structure-meaning-learning.git
cd structure-meaning-learning

uv venv --python 3.9 .venv
source .venv/bin/activate # Rerun this every time

uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
uv pip install numpy==1.23.5 protobuf==3.20.1
uv pip install -r requirements.txt

git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git
uv pip install -e ./pytorch-struct
```

# Extra Notes

When running without GPU, there are errors involving `lengths` as a list instead of a PyTorch tensor.

This was fixed by converting `lengths` to tensors in the collate functions in `vc-pcfg/vpcfg/as_dataloader.py`.