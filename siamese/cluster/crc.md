# CRC Guide

This file documents the current recommended workflow for running the Siamese
signature project on the Notre Dame CRC.

## Goal

Use CRC for full Bengali/Hindi training on one general GPU, and keep evaluation
local on your Mac unless the dataset and checkpoints only exist on CRC.

## Storage Layout

Do not build the workflow around `/scratch365`. The CRC docs note that
`/scratch365` is retiring in June 2026. Use home storage under `/users/<netid>`
for now.

Recommended layout on CRC:

```text
~/signature-project/
  repo/
  data/
    cedar-bhsig260/
      BHSig260-Bengali/
      BHSig260-Hindi/
  logs/
  runs/
```

## 1. Connect To CRC

From your Mac:

```bash
ssh your_netid@crcfe02.crc.nd.edu
```

Optional VS Code:

1. Install the `Remote - SSH` extension.
2. Add an SSH config entry like:

```sshconfig
Host crc
    HostName crcfe02.crc.nd.edu
    User your_netid
```

3. Connect to `crc` from VS Code.
4. If VS Code keeps asking for your password repeatedly, enable the Remote SSH
   setting `Lockfiles In Tmp`.

## 2. Create Project Directories

On CRC:

```bash
mkdir -p ~/signature-project/{data,logs,runs}
cd ~/signature-project
```

## 3. Clone The Repo

Use SSH so GitHub does not prompt for HTTPS credentials:

```bash
git clone git@github.com:smondal13/CSE60868-signature-project.git repo
cd ~/signature-project/repo
git fetch --all
git switch siamese-shuvo
git status
```

The expected result is a clean working tree on branch `siamese-shuvo`.

## 4. Initialize Conda On CRC

If this is your first time using Conda on CRC:

```bash
module load conda
conda init
source ~/.bashrc
module unload conda
conda info
```

After the initial setup, CRC recommends that you do not load the `conda` module
inside your batch jobs.

## 5. Create The Environment

From the repo root:

```bash
cd ~/signature-project/repo
conda env create -f environment.yml
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

Activate it:

```bash
conda activate machine-learning
```

Install the package in editable mode:

```bash
pip install -e . --no-deps --no-build-isolation
```

The `--no-build-isolation` flag matters on CRC because otherwise `pip` may try
to download build dependencies like `setuptools` from the internet during the
editable install.

## 6. Copy The Bengali And Hindi Data To CRC

Run these commands from your local Mac terminal, not from CRC:

```bash
rsync -avP "/Users/smondal/Library/CloudStorage/GoogleDrive-smondal@nd.edu/My Drive/1. ND/1. PhD courses/Spring 26/Project/cedar-bhsig260/BHSig260-Bengali" smondal@crcfe02.crc.nd.edu:~/signature-project/data/cedar-bhsig260/
```

```bash
rsync -avP "/Users/smondal/Library/CloudStorage/GoogleDrive-smondal@nd.edu/My Drive/1. ND/1. PhD courses/Spring 26/Project/cedar-bhsig260/BHSig260-Hindi" smondal@crcfe02.crc.nd.edu:~/signature-project/data/cedar-bhsig260/
```

Then confirm on CRC:

```bash
ls ~/signature-project/data/cedar-bhsig260
```

You should see:

- `BHSig260-Bengali`
- `BHSig260-Hindi`

## 7. Generate Manifests On CRC

Set the dataset root and generate the Bengali/Hindi manifests:

```bash
cd ~/signature-project/repo
conda activate machine-learning
export SIGNATURE_DATA_ROOT=$HOME/signature-project/data/cedar-bhsig260
python -m signature_siamese.prepare_data
```

This regenerates:

- `siamese/manifests/bhsig260_index_raw.csv`
- `siamese/manifests/bhsig260_small_manifest.csv`
- `siamese/manifests/bhsig260_manifest.csv`

## 8. CPU Sanity Check

Before using a GPU:

```bash
cd ~/signature-project/repo
conda activate machine-learning
python -c "import torch; import signature_siamese; print(torch.__version__)"
python -c "from pathlib import Path; from signature_siamese.data.datasets import SignatureDataset; ds = SignatureDataset(Path('siamese/manifests/bhsig260_small_manifest.csv'), split='train'); print(len(ds))"
```

## 9. Interactive GPU Smoke Test

Start an interactive general GPU session:

```bash
qrsh -q gpu -l gpu_card=1 -pe smp 4
```

Inside the GPU session:

```bash
source ~/.bashrc
conda activate machine-learning
cd ~/signature-project/repo
export SIGNATURE_DATA_ROOT=$HOME/signature-project/data/cedar-bhsig260
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

If CUDA is visible, run a tiny import or debug command before submitting a long
job.

## 10. Configure Full Training

Before submitting the real training run, confirm in
`siamese/src/signature_siamese/train.py` that:

- `RUN_PROFILE = "full"`
- `DEVICE = "auto"`

The current code will choose CUDA automatically on CRC when available.

## 11. Create A CRC Batch Script

Save this as `~/signature-project/repo/siamese/cluster/train_crc.job`:

```bash
#!/bin/bash
#$ -M smondal@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu_card=1
#$ -l h_rt=24:00:00
#$ -pe smp 4
#$ -N sig_train
#$ -o /users/smondal/signature-project/logs/sig_train.$JOB_ID.out
#$ -e /users/smondal/signature-project/logs/sig_train.$JOB_ID.err

source ~/.bashrc
conda activate machine-learning

cd /users/smondal/signature-project/repo
export SIGNATURE_DATA_ROOT=/users/smondal/signature-project/data/cedar-bhsig260

python -m signature_siamese.train
```

Then submit:

```bash
mkdir -p ~/signature-project/logs
qsub ~/signature-project/repo/siamese/cluster/train_crc.job
```

## 12. Monitor The Job

```bash
qstat -u smondal
```

Optional:

```bash
qstat -j <job_id>
free_nodes.sh -G
```

Common queue states:

- `qw`: waiting in queue
- `r`: running

## 13. Evaluate Locally

Recommended workflow:

1. Train on CRC.
2. Copy the best checkpoint back to your Mac.
3. Evaluate and plot locally.

Example copy-back command from your Mac:

```bash
rsync -avP smondal@crcfe02.crc.nd.edu:~/signature-project/repo/siamese/runs/<run_name>/checkpoints/best.pt ~/Downloads/
```

Then evaluate locally using your existing `machine-learning` environment and the
local Bengali/Hindi dataset.

## 14. Hyperparameter Optimization

Do not start full hyperparameter optimization yet.

Recommended order:

1. Get one clean full CRC training run working end to end.
2. Confirm runtime, checkpointing, and logs are stable.
3. Evaluate locally.
4. Only then start a small search over a few parameters.

Best first parameters to tune later:

- learning rate
- contrastive margin
- writers per batch
- samples per writer
- hard negatives per positive

## CRC References

- Quick start: <https://docs.crc.nd.edu/new_user/quick_start.html>
- Connecting to CRC: <https://docs.crc.nd.edu/new_user/connecting_to_crc.html>
- Batch jobs: <https://docs.crc.nd.edu/new_user/submitting_batch_jobs.html>
- GPU usage: <https://docs.crc.nd.edu/resources/gpu.html>
- Conda: <https://docs.crc.nd.edu/popular_modules/conda.html>
- FAQ: <https://docs.crc.nd.edu/faq/faq.html>
