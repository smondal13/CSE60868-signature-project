# CRC Job Templates (Siamese Track)

This folder contains starter templates for running the Siamese pipeline on a single GPU.
The training/evaluation scripts now use top-level variables (no argparse), so set options
in `siamese-shuvo/src/signature_siamese/*.py` before submitting jobs.

## 1. Prerequisites
- Repository is available on CRC filesystem.
- Conda environment `machine-learning` is installed on CRC and includes PyTorch + TensorBoard.
- In `siamese-shuvo/src/signature_siamese/train.py`, set:
  - `RUN_PROFILE = "full"`
  - `DEVICE = "cuda"`

## 2. UGE/SGE (`qsub`) path
1. Update resource flags in `siamese-shuvo/cluster/train_siamese_qsub.sh` if your queue uses different GPU labels.
2. Submit:
```bash
qsub siamese-shuvo/cluster/train_siamese_qsub.sh
```
3. Monitor:
```bash
qstat
```
4. Inspect logs:
```bash
tail -f siamese-shuvo/logs/siamese_sig.<job_id>.log
```

## 3. HTCondor path
1. Edit `arguments` in `siamese-shuvo/cluster/train_siamese.condor`:
   - first argument: repo path on CRC
2. Submit:
```bash
condor_submit siamese-shuvo/cluster/train_siamese.condor
```
3. Monitor:
```bash
condor_q
```
4. Inspect logs:
```bash
tail -f siamese-shuvo/logs/condor_siamese.<cluster_id>.<proc_id>.out
```

## 4. Common failure points (first-time CRC users)
- Missing conda activation script path.
- Queue-specific GPU resource names differ from template.
- Manifest path mismatch between login node and compute node.
- Permissions on `siamese-shuvo/logs/` and `siamese-shuvo/runs/` directories.
