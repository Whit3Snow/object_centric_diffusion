# Environment setup on the B200 cluster

The upstream README targets Ubuntu 22 + RTX 4090 with `python 3.8 / torch 2.3.1 / cu121`.
**That combination cannot run here**: cu121 builds top out at `sm_90`, and these
are B200 (`sm_100`) GPUs. What follows is the working recipe, with every
deviation from the README called out.

Everything lives on NAS, so worker pods see the same paths as the login node.

| | path |
|---|---|
| conda env | `/home/nas_main/hyojinjang/miniconda3/envs/spot` |
| CoppeliaSim 4.1.0 | `/home/nas_main/hyojinjang/CoppeliaSim` |
| PyRep / YARR / RLBench sources | `/home/nas_main/hyojinjang/repos/2026/spot_deps` |

## Use it

```bash
source scripts/spot_env.sh     # conda activate + CoppeliaSim + CUDA + LD_LIBRARY_PATH
```

Training goes through sbatch, never the login node:

```bash
sbatch scripts/sbatch_train_icl.sh rlbench_icl_multi other_episode
```

## Deviations from the README (and why)

| README | here | why |
|---|---|---|
| python 3.8 | **3.10** | cu128 wheels and numba 0.58 need ≥3.9 |
| torch 2.3.1 + cu121 | **2.8.0 + cu128** | cu121 has no `sm_100`; B200 would fail with "no kernel image is available" |
| `conda install pytorch3d -c pytorch3d` | **source build of pytorch3d 0.7.9** | no prebuilt channel package for torch 2.8 / cu128 |
| `conda install cuda -c nvidia/label/cuda-12.1.0` | **`cuda-nvcc` etc. from `nvidia/label/cuda-12.8.0`** | must match torch's CUDA; kept inside the env so it survives on workers |
| `sudo apt install libboost-all-dev` | **conda `boost`** | no sudo on this cluster |
| `numpy==1.23.5`, `scipy==1.10.1` (fp_requirements) | **numpy 1.26.4, scipy 1.11.4** | `moviepy` (a dp3 requirement) forces numpy ≥1.25; scipy 1.10.1 is not ABI-compatible with it |
| — | **tensorboard pinned to 2.19.0** | YARR pulls tensorboard 2.21, which needs protobuf ≥6.31 while wandb 0.17.4 needs <6 |

## Traps hit along the way

* **`CUDA_HOME` must be the conda env, not `/usr/local/cuda-12.8`.** Worker pods
  are driver-only; anything linked against `/usr/local/cuda-*` fails there with
  `libcudart.so.12: cannot open shared object file`. `scripts/spot_env.sh` sets
  `CUDA_HOME=$CONDA_PREFIX`.
* **conda's CUDA headers land in `$CONDA_PREFIX/targets/x86_64-linux/include`**,
  not `$CONDA_PREFIX/include`, so torch's `cpp_extension` can't find
  `cuda_runtime.h`. They are symlinked into `$CONDA_PREFIX/include`; redo it
  after installing more CUDA packages:

  ```bash
  for f in $CONDA_PREFIX/targets/x86_64-linux/include/*; do
      ln -s "$f" "$CONDA_PREFIX/include/$(basename $f)" 2>/dev/null
  done
  ```
* **nvdiffrast needs `--no-build-isolation`** plus the CUDA math headers
  (`libcusparse-dev`, `libcublas-dev`, `libcusolver-dev`, `libcurand-dev`,
  `libcufft-dev`).
* **`libGL.so.1` is missing system-wide** — this node ships only
  `libGLX_nvidia`, no libglvnd/Mesa. `open3d` dlopens `libGL.so.1`, and
  `simple_dp3` imports open3d through `utils/vis_utils.py`, so **even training**
  fails without it. Fixed with conda-forge `mesalib`, exposed through
  `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` in `scripts/spot_env.sh`.
* **RLBench's `setup.py` omits `rlbench.action_modes`** (upstream README's
  troubleshooting note). Patched in the local clone before `pip install .`.
* **Never `pip install --force-reinstall --no-deps numpy`** into this env: it
  leaves a hybrid package tree and torch then dies with
  `TypeError: expected np.ndarray (got numpy.ndarray)`. Uninstall numpy fully,
  delete `site-packages/numpy*`, then reinstall.
* **CoppeliaSim needs an X display.** Headless launch works with the conda
  `Xvfb`:

  ```bash
  Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99
  ```

  Verified: `PyRep().launch(task_design.ttt, headless=True)` starts and steps.

## Verified

* `torch 2.8.0+cu128`, arch list includes `sm_100`
* full SPOT + ICL import chain (pytorch3d, nvdiffrast, warp, open3d, mycpp)
* FoundationPose `mycpp` extension built and loaded
* `pyrep` / `rlbench` / `yarr` import; CoppeliaSim launches headless under Xvfb
* the ICL test suite (dataset prompt logic, `compute_loss` with gradient into
  the prompt encoder, `predict_action` in both training and rollout form, EMA
  deepcopy) passes
* 5 training steps + `predict_action` of `PromptSimpleDP3` on a B200

## Still needed before training

The **PerAct pre-generated dataset** (train 100 / val 25 / test 25) and the
**RLBench object meshes** are manual Google Drive downloads. Once they are on
NAS, point `--peract_demo_dir` / `--save_path` in
`scripts/gen_demonstration_rlbench.sh` at them, generate the zarr, and set
`dataset.root_dir` in the task configs.
