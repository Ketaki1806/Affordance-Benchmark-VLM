# Affordance Benchmark

Caption generation pipeline using Qwen with affordance-focused prompts.

## Project layout

```
configs/          # YAML config, prompt template, logging
src/              # Pipeline, model, caption generation
scripts/          # Setup and run helpers
environment.yml   # Python 3.11 base (conda-forge)
requirements-gpu.txt
```

## Local setup (Windows)

```powershell
.\scripts\setup_env.ps1
conda activate affordance_benchmark
.\scripts\run_pipeline.ps1
```

## LST cluster (HTCondor)

The LST cluster uses **[HTCondor](https://wiki.lst.uni-saarland.de/doku.php?id=user:cluster:a_condor)**, not Slurm. There is no `sbatch` — use `condor_submit` on the **submit** node.

| Node | Hostname | Purpose |
|------|----------|---------|
| Login | `login.lst.uni-saarland.de` | Edit files, light setup |
| Submit | `ssh submit` | Submit HTCondor jobs |
| Execute | (assigned by Condor) | Runs your job (GPU, more RAM) |

### 1. Setup on login node

```bash
ssh kahadnurkar@login.lst.uni-saarland.de
cd ~/Affordance_Benchmark

mkdir -p ~/tmp ~/.cache/pip
export TMUX_TMPDIR=~/tmp TMPDIR=~/tmp TEMP=~/tmp TMP=~/tmp

tmux new -s setup
bash scripts/install_micromamba.sh
bash scripts/setup.sh
bash scripts/install_deps.sh    # micromamba for PyTorch; or use Condor step 2
```

### 2. Install deps on an execution node (recommended)

```bash
ssh submit
cd ~/Affordance_Benchmark
bash scripts/condor_submit_install.sh

condor_q                          # your jobs
tail -f artifacts/logs/install-deps.<ClusterId>.out
```

### 3. Run the pipeline on GPU

```bash
ssh submit
cd ~/Affordance_Benchmark
bash scripts/condor_submit_pipeline.sh

condor_q
tail -f artifacts/logs/pipeline.<ClusterId>.out
```

### Useful HTCondor commands (on submit node)

```bash
condor_q              # queue
condor_q -better-analyze <ClusterId>   # why job is idle
condor_rm <ClusterId> # cancel
```

### Each new SSH session

```bash
cd ~/Affordance_Benchmark
source scripts/activate_env.sh
```

### Reset / wipe install artifacts

Remove micromamba, caches, temp files, and broken `~/.bashrc` entries (keeps the repo):

```bash
bash scripts/cluster_cleanup.sh --all
```

Also delete the project directory:

```bash
bash scripts/cluster_cleanup.sh --all --project
```

## Notes

- Login node `/tmp` is on a full local disk — scripts use `~/tmp` on nethome.
- Do not `pip install torch` on the login node (OOM → `Killed`). Use `bash scripts/install_deps.sh` (micromamba) or Condor.
- Log into the [LST Condor wiki](https://wiki.lst.uni-saarland.de/doku.php?id=user:cluster:a_condor) for site-specific GPU requirements.
