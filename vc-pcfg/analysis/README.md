# Category Analysis

All category analysis can be found in the `vc-pcfg/analysis/` directory.

## Current Scripts

* `export_word_categories.py` exports a single CSV containing one row per token with its induced preterminal category and sentence context
* `analyze_word_categories.py` comprehensively analyzes a single exported category CSV by category

## Recommended Layout

```text
vc-pcfg/analysis/outputs/
├── category_analysis/      <-- store output of analyze_word_categories.py here
│   └── s91-e5-c60/         <-- example analysis; named with seed, # epochs, and # preterminal categories
├── llm_response/           <-- store LLM qualitative response here
│   ├── s91-e5-c60/
│   ├── PROMPT.txt          <-- LLM qualitative analysis prompt sent with each query
│   └── SCHEMA.json         <-- required LLM output schema
├── model_checkpoints/      <-- store model checkpoints here after running them on the cluster
│   └── s91-e5-c60/
└── word_csvs/              <-- store exported word categories here
    └── s91-e5-c60.csv
```

## Setup for Training a Model Locally and Running `export_word_categories.py`

```bash
# Clone repo
git clone https://github.com/evaportelance/structure-meaning-learning.git
cd structure-meaning-learning

# Setup virtual environment
uv venv --python 3.9 .venv
source .venv/bin/activate # Rerun this every time

# Install dependencies
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
uv pip install numpy==1.23.5 protobuf==3.20.1
uv pip install -r requirements.txt
git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git
uv pip install -e ./pytorch-struct
```

## Running `export_word_categories.py`

```bash
source .venv/bin/activate
cd vc-pcfg

# Either initialize a model locally for testing, or copy model a model checkpoint from the cluster
# Below is how to initialize a model locally:
python as_train.py \
    --data_path ../preprocessed-data/abstractscenes \
    --logger_name analysis/outputs/model_checkpoints/random_init_test \
    --tiny \ # Not necessary
    --visual_mode \
    --init_only \
    --encoder_file \
    "all_as-resn-50.npy"

# Then run export:
python -m analysis.export_word_categories \
    --model_init analysis/outputs/model_checkpoints/random_init_test/checkpoints/checkpoint.pth.tar \
    --data_path ../preprocessed-data/abstractscenes \
    --output_path analysis/outputs/random_word_categories.csv \
    --use_mean_z # For reproducibility
```

## Running `analyze_word_categories.py`

```bash
uv venv .venv-analysis --python 3.12.3
source .venv-analysis/bin/activate

# Install dependencies if necessary
uv pip install numpy pandas matplotlib spacy click scipy
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

cd vc-pcfg

python -m analysis.analyze_word_categories \
    --input analysis/outputs/s91-e5.csv \
    --output-dir analysis/outputs/category_analysis/s91-e5 \
    --spacy-model en_core_web_sm
```

# Training a Model on the Cluster

In `$HOME`, clone `https://github.com/iangif/grammar-induction-categories`

`cd` into cloned repo

## Create `requirements-lock.txt`

Create `requirements-lock.txt` with:

```text
numpy==1.26.4+computecanada
protobuf==3.20.1
transformers
datasets
benepar==0.2.0
nltk
```

## Set Up the Environment

Run the following:

```bash
# Disable any running venvs or loaded modules
deactivate 2>/dev/null || true
module purge

# Load narval modules
module load python/3.10
module load StdEnv/2023
module load gcc arrow/21.0.0

# Create virtual environment
mkdir -p ~/venvs
python -m venv ~/venvs/graminduct
source ~/venvs/graminduct/bin/activate

# Install requirements
python -m pip install -r requirements-lock.txt

# Install torch
python -m pip install \
  torch==2.5.0 \
  torchvision==0.20.0 \
  torchaudio==2.5.0

# Install pytorch-struct
cd "$HOME"
git clone \
  --branch infer_pos_tag \
  https://github.com/zhaoyanpeng/pytorch-struct.git
python -m pip install -e "$HOME/pytorch-struct"
```

## Create the Cluster Script

Make `scripts/` directory, and make `smoke_en_joint.sh` with contents the same as `vc-pcfg/analysis/cluster_script.sh`

You can edit the following:

* `NUM_EPOCHS`
* `T_STATES` (number of preterminal categories)
* `SEED`

## Make the Script Executable

```bash
chmod +x scripts/smoke_en_joint.sh
```

## Run the Job

```bash
sbatch scripts/smoke_en_joint.sh
```

## View the Job and Job ID

```bash
sq
```

## View the Status of the Current Run

View the status of the current run with the correct job ID:

```bash
tail -f slurm-en_smoke-65480625.out
```

## Retrieve the Model

When complete, view the model in `runs/`, and copy it to your local machine.
