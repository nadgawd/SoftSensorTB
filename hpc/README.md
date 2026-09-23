# Running the LLM on the IITD HPC

Replaces the cloud API with a vLLM server on a GPU compute node. The backend
still runs on your laptop and reaches the cluster over an SSH tunnel.

All settings live in one file, [`env.sh`](env.sh). Source it before anything
else; nothing else hardcodes a path.

```bash
source hpc/env.sh
```

## Most of this cluster's GPUs cannot run vLLM

This is the single most important thing to know. A plain
`-l select=1:ncpus=8:ngpus=1` lands you on a `khas` node, which is a **Tesla
K40m**: compute capability 3.5, 11 GB, driver 470. vLLM requires 7.0 or higher
and modern PyTorch is not even built for Kepler. There is no configuration that
makes it work, and 74 of the 131 GPU nodes are these.

PBS exposes no `gpu_model` resource here, but the CPU generation maps onto the
GPU generation, so `centos` is a usable selector:

| Prefix | `centos`   | Nodes | GPU                            |
|--------|------------|-------|--------------------------------|
| `khas` | `haswell`  | 74    | Tesla K40m — **unusable**      |
| `vsky` | `skylake`  | 36    | V100-class                     |
| `aice` | `icelake`  | 16    | **A100-PCIE-40GB**, driver 590 |
| `scai` | `amdepyc`  | 3     | 8 GPU, 1 TB RAM (restricted)   |

`serve_llm.pbs` therefore requests `centos=icelake`. Keep the CPU chunk small —
asking for 8 cores queued indefinitely behind *"Insufficient amount of
resource: ncpus"* while GPUs sat idle; 4 cores schedules immediately and
inference is GPU-bound anyway.

## Three facts that explain most of the friction

**Scratch is not `/scratch/<username>`.** It is carved up by department and
programme: `/scratch/chemical/dual/ch7221493/mtp`. Scripts that guess
`/scratch/$(whoami)` produce a path you have no permission to create. Set
`HPC_DEPT` / `HPC_PROGRAMME` in `env.sh`, or export `SCRATCH_DIR` directly.

**The login nodes reach the internet through the proxy with no portal login.**
`env.sh` exports the proxy variables for you. Compute nodes are stricter and
need `proxy_login.sh`, but nothing in this workflow requires that, because
downloads happen on the login node. Forgetting the proxy makes huggingface_hub
hang silently rather than fail, which is why `env.sh` now enables it by default.

**Serving needs no internet at all.** Weights sit on scratch and
`HF_HUB_OFFLINE=1` stops vLLM phoning home. So the serve job is an ordinary
batch `qsub`, with no lynx window to keep alive.

## One-time setup

Run on a **login node**. Needs no GPU, downloads ~4 GB of wheels.

```bash
cd ~/mtp
source hpc/env.sh
bash hpc/setup_scratch.sh    # conda env on scratch (~20 min, ~10 GB)
bash hpc/fetch_model.sh      # Qwen3.5-27B-GPTQ-Int4, 29 GB
MODEL_PROFILE=qwen2.5-7b bash hpc/fetch_model.sh   # optional 7B fallback, 16 GB
```

Run both under `tmux` so a dropped SSH session doesn't leave a half-built
environment behind.

The env is created with `conda create --prefix` on scratch, not `--name`: a
named env goes to `~/.conda/envs` and torch alone would exhaust the home quota.
`CONDA_PKGS_DIRS` and `PIP_CACHE_DIR` are redirected for the same reason.

Scripts invoke `$ENV_PY` (an absolute interpreter path) rather than activating
the env. In a non-interactive shell `conda activate` can return success while
leaving `PATH` untouched, so `pip` silently resolves to the read-only system
conda and dies with `EACCES`.

### Two path traps worth knowing

The `apps/miniconda` module exports
`PYTHONPATH=<base>/lib/python3.12/site-packages`. That leaks into *every*
interpreter afterwards — including the 3.10 env — so pip finds base packages,
tries to uninstall root-owned files, and fails with `EACCES`. It is also an ABI
mismatch. `load_conda` in `env.sh` unsets it.

Old `pip install --user` packages in `~/.local` similarly shadow the env;
`PYTHONNOUSERSITE=1` disables that path. To reclaim the quota:
`rm -rf ~/.local/lib/python3.*/site-packages`.

## Daily workflow

From the project root, one command does all of it:

```bash
./run.sh              # GPU job + tunnel + backend + frontend
./run.sh --stop       # tear down local processes (PBS job survives)
./run.sh --no-cluster # laptop only; chat uses the cloud keys in .env
```

It starts the model profile named by `LOCAL_LLM_EXECUTION_MODEL` in `.env`,
reusing an already-running job for that profile (`vllm_qwen3527b` or
`vllm_qwen257b`) rather than submitting a second one. It warns, with a `qdel`
command, about LLM jobs for any other profile that are still holding a GPU. It
reads the compute node from PBS instead of the buffered job log, and waits for
the model to finish loading before starting the backend. Ctrl-C stops the local
processes and leaves the GPU job up, since that costs a few minutes to restart.

<details>
<summary>The same thing by hand</summary>

```bash
# 1. on the cluster (default profile, qwen3.5-27b)
cd ~/mtp && qsub hpc/serve_llm.pbs
qstat -u "$USER"
grep "Running on node" hpc/logs/vllm_qwen3527b.out  # -> aice006

# 2. on your laptop
COMPUTE_NODE=aice006 bash hpc/tunnel.sh             # leave open

# 3. another laptop terminal
bash hpc/check_server.sh localhost 8888
source .venv_run/bin/activate && uvicorn backend.main:app --reload --port 8000
cd frontend && npm run dev
```
</details>

One subtlety worth knowing if you tunnel by hand: when a ControlMaster socket
exists, a `-L` forward belongs to the *master* process, not the client that
asked for it. Killing the client leaves the port bound, and your next tunnel
dies silently. Release it with
`ssh -O cancel -L 8888:<node>:8000 iitd`.

Startup takes one to three minutes: about 20 s to load the weights, then
torch.compile (about 50 s the first time, cached on scratch afterwards),
profiling and KV-cache setup. The server is silent until it is ready, and PBS
buffers job output, so poll `curl http://<node>:8000/v1/models` from the login
node rather than watching the log.

### Passwordless repeat logins

Add this to your laptop's `~/.ssh/config`, authenticate once with `ssh iitd`,
and every later `ssh`, `rsync`, and tunnel reuses that connection:

```
Host iitd
    HostName hpc.iitd.ac.in
    User ch7221493
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 8h
```

Closing that first window tears down the socket and everything reverts to
prompting for a password.

Then point the backend at it:

```
LLM_MODE=local_first
LOCAL_LLM_BASE_URL=http://localhost:8888/v1
LOCAL_LLM_EXECUTION_MODEL=qwen3.5-27b
LOCAL_LLM_KNOWLEDGE_MODEL=qwen3.5-27b
LOCAL_LLM_ROUTER_MODEL=qwen3.5-27b
LOCAL_LLM_THINKING=1
```

To fall back to the 7B, set the three model lines to `qwen2.5-7b`, remove
`LOCAL_LLM_THINKING` and re-run `./run.sh`; it starts the matching job.

`local_first` is the right mode while the cluster is the weak link: the tunnel
dies when your walltime expires, and this falls back to a cloud key rather than
breaking chat mid-session.

`check_server.sh` does more than ping `/v1/models` — it sends a real tool-call
request, because a server that chats perfectly can still fail to emit tool calls
when `--tool-call-parser` doesn't match the model, and that failure is invisible
until the agent silently stops calling tools.

## Which model

`env.sh` defines two profiles. The profile name is also the name the model is
served under, which is what `.env` refers to.

| | `qwen3.5-27b` (default) | `qwen2.5-7b` (fallback) |
|---|---|---|
| Checkpoint | `Qwen/Qwen3.5-27B-GPTQ-Int4` | `Qwen/Qwen2.5-7B-Instruct` |
| GPU memory for weights | 26.7 GiB | ~15 GiB |
| Context | 64k (KV cache holds ~137k tokens) | 32k |
| Decode speed, one stream | ~42 tok/s compiled | fast either way |
| Tool-call parser | `qwen3_coder` | `hermes` |
| Thinking mode | yes, toggled per request | no |

The 7B was the bring-up model: unquantized, quick to download and easy to
debug the firewall, PBS, tunnel and tool parsing against. The 27B is much
better at the agent's actual work, which is choosing the right tool from ~25,
filling arguments with real column names and chaining several calls.

The 27B's flags are explained at the bottom of `serve_llm.pbs`. Two are easy to
get wrong. It needs `--quantization gptq_marlin`, not the model card's
`moe_wna16`, which only quantizes MoE experts and so loads this dense model's
MLPs in bf16 and runs out of memory. And it compiles by default: eager mode
decodes at ~12 tok/s against ~42 compiled.

**Thinking.** Qwen3.5 can reason before answering. The server starts with
thinking off. The chat's "Think" toggle turns it on for a request; the backend
then sends `enable_thinking: true` with the model card's sampling settings.
The reasoning streams into the activity log, separate from the answer. It makes
replies noticeably slower, so it is best kept for hard questions.

**Other GPUs.** Compute capability, printed at the top of every serve log,
decides what works. 8.0 or higher (A100 and newer) is needed for the GPTQ
Marlin kernels the 27B uses. On a 7.0 V100, use the 7B profile.

Adding a profile means a new entry in the `case` blocks of `env.sh` and
`serve_llm.pbs`, then `MODEL_PROFILE=<name> bash hpc/fetch_model.sh`.

## On fine-tuning

Probably unnecessary. The router settles most intents in a regex layer that
never calls a model, with about forty phrasings pinned in
`tests/test_router.py`, including cases where a theory question reuses an action
verb. Only genuinely ambiguous turns reach the model, and the served model
handles those.

`train_lora.pbs` is there for when logged conversations show the served model
mishandling real tool arguments — use those turns as the training set. Note that
you cannot LoRA a quantized checkpoint: train against the unquantized base with
`--use_4bit` and serve the merged result.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `cd: /scratch/ch7221493: No such file` | Missing dept/programme prefix; see `SCRATCH_DIR` in `env.sh` |
| Job stuck `Insufficient amount of resource: ncpus` | CPU chunk too big for the busy A100 nodes; 4 is enough |
| vLLM: `no kernel image is available` | You landed on a K40m; the request lost `centos=icelake` |
| `PermissionError: 'ninja'` / `'nvcc'` | Env `bin/` not on `PATH`; `serve_llm.pbs` adds it |
| `CUDA compiler and CUDA toolkit headers are incompatible` | flashinfer JIT; `VLLM_USE_FLASHINFER_SAMPLER=0` |
| pip `EACCES` on `/home/apps/miniconda3/...` | Module's `PYTHONPATH` leak; `load_conda` unsets it |
| Download hangs with no output | Proxy not exported — re-source `env.sh` |
| pip `302 Moved Temporarily` | Compute node not authenticated — `bash hpc/proxy_login.sh` |
| Model loads, never calls tools | Wrong `--tool-call-parser`; Qwen3.5 needs `qwen3_coder`, Qwen2.5 `hermes` |
| `<think>` text shows up in answers | `--reasoning-parser qwen3` missing |
| 404 on every chat turn | `LOCAL_LLM_*_MODEL` doesn't match the served profile name |
| Tunnel open, connection refused | Job ended — `qstat -u $USER`, re-`qsub`, node name changed |
| 27B OOM while creating weights | `--quantization moe_wna16` instead of `gptq_marlin` |
| CUDA OOM at load | Add `-v MAX_MODEL_LEN=32768` to the `qsub`, then lower `--gpu-memory-utilization` |
| Job queued on `Insufficient amount of resource: ngpus` | No free A100; an old LLM job of yours may hold one (`run.sh` lists them) |

Scratch is not backed up and is purged after a period of inactivity. Model
weights are cheap to re-download; anything you care about belongs in `$HOME` or
off the cluster. Confirm the exact window with `hpchelp@iitd.ac.in`.
