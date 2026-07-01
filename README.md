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

## LST cluster setup

SSH in, clone the repo, then:

```bash
cd ~/Affordance_Benchmark

# Redirect temp/cache off the full login-node /tmp
mkdir -p ~/tmp ~/.cache/pip
export TMUX_TMPDIR=~/tmp TMPDIR=~/tmp TEMP=~/tmp TMP=~/tmp

# Use tmux so install survives disconnect
tmux new -s setup
bash scripts/install_micromamba.sh
bash scripts/setup.sh
sbatch scripts/install_deps.slurm
tail -f artifacts/logs/install-deps-<JOBID>.out
# Ctrl+B, D to detach
```

Each new session:

```bash
cd ~/Affordance_Benchmark
source scripts/activate_env.sh
```

Submit a GPU job (from the submit node once Slurm is available):

```bash
sbatch scripts/submit_gpu_job.slurm
```

## Notes

- On LST login nodes, always use `~/tmp` instead of `/tmp` (local disk is full).
- Heavy work belongs on GPU compute nodes, not the login node.
