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
  Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -noreset & export DISPLAY=:99
  ```

  GL itself is fine on this node - a GLX context created against that Xvfb
  reports `NVIDIA B200 / OpenGL 4.6 / direct rendering`.
* **`QT_QPA_PLATFORM=offscreen` breaks CoppeliaSim rendering.** The offscreen
  platform plugin creates no GL context, so the first vision-sensor render
  segfaults (`QOpenGLFramebufferObject` -> `QOpenGLContext::shareGroup` on a
  null context). Use `xcb` under Xvfb; `scripts/spot_env.sh` does.
  A bare `PyRep().launch(); pr.step()` smoke test does *not* catch this - it
  never renders a vision sensor.
* **The `opengl3` renderer segfaults headless anyway.** Even with a working GL
  context, `libsimExtOpenGL3Renderer.so` crashes in the same place. Pass
  `--renderer=opengl` (the gen script defaults to it via `RENDERER`). Nothing
  is lost: the zarr stores only object poses - the `img` / `point_cloud`
  dataset writes are commented out upstream in `utils/collect_utils.py`.
* **SPOT needs a patched RLBench task** (not mentioned in the README).
  `env_rlbench_peract/utils/rlbench_utils.py::_get_misc` reads
  `self.task._chosen_pillar_name`, which stock `insert_onto_square_peg.py`
  never sets - it keeps `chosen_pillar` as a local. Add to `init_episode`:

  ```python
  chosen_pillar = np.random.choice(spokes)
  self._chosen_pillar_name = chosen_pillar.get_name()   # <- add
  ```

  This is correct rather than a guess: `reset_to_demo()` runs
  `demo.restore_state()` (which restores the numpy RNG state) *before*
  `init_episode()`, so `np.random.choice` re-picks the demo's own pillar.
  Of the task attributes SPOT expects (`_chosen_pillar_name`, `_cups`,
  `_spokes`), only this one is missing from stock RLBench.

## Verified

* `torch 2.8.0+cu128`, arch list includes `sm_100`
* full SPOT + ICL import chain (pytorch3d, nvdiffrast, warp, open3d, mycpp)
* FoundationPose `mycpp` extension built and loaded
* `pyrep` / `rlbench` / `yarr` import; CoppeliaSim launches headless under Xvfb
* zarr generation end to end: `insert_onto_square_peg` test split, 2 episodes
  -> `state (22, 7)` with unit quaternions, loads back through `ReplayBuffer`
* the ICL test suite (dataset prompt logic, `compute_loss` with gradient into
  the prompt encoder, `predict_action` in both training and rollout form, EMA
  deepcopy) passes
* 5 training steps + `predict_action` of `PromptSimpleDP3` on a B200

## Datasets

Both downloads are scripted - no manual Google Drive clicking needed.

```bash
# PerAct demonstrations: 13 SPOT tasks x {train,test} = 26 archives, ~77 GB
python tools/download_peract.py --out <DATA>/peract/raw --jobs 3 --extract

# RLBench object meshes (~23 MB), needed for FoundationPose at eval time
gdown --folder -O <DATA>/mesh "https://drive.google.com/drive/folders/1bupiLa2akr2sytb7jcULnU_6ed4OOgkw"
```

Then point `--peract_demo_dir` / `--save_path` in
`scripts/gen_demonstration_rlbench.sh` at `<DATA>/peract/raw` and the zarr
output dir, and set `dataset.root_dir` in the task configs plus
`pose_estimation.mesh_dir` to `<DATA>/mesh`.

### Generation traps

* **`--disable_cameras` is required on workers** (see the Traps section): with
  it the whole generation is CPU-only and needs no X server. Submit one job per
  task, `sbatch scripts/sbatch_gen_demonstration.sh <task>`; each takes roughly
  5-70 min (~8 min for a typical task). A split whose zarr exists is skipped,
  so a preempted job can simply be resubmitted.
* **Short episodes crashed `turn_tap`.** `utils/collect_utils.py` guarded
  `progress_binary[-4..-6]` by length but not `[-2]` / `[-3]`, so a 2-keyframe
  episode raised `IndexError` and took the whole task down. Fixed.
* **Some demos never replay.** RLBench fails to re-plan a few episodes
  (`Bad demo... not successful`, `Could not get a path for waypoint 0`); the
  collector retries 10x then skips. Result: `light_bulb_in` 99/100,
  `turn_tap` 99/100, `put_groceries_in_cupboard` 90/100. The RNG state comes
  from the recorded demo, so retrying reproduces the same failures.
* **Multi-stage tasks produce more episodes than demos** - one demo becomes one
  episode per stage (`stack_cups` 100 demos -> 200, `stack_blocks` -> 294,
  `place_cups` -> 208). Expected, not an error.

### Google Drive traps

* The PerAct folder is an **old-style Drive folder with a `resourceKey`**.
  `gdown --folder` does not send it and just returns `401`. `download_peract.py`
  lists the folder through the Drive v3 API instead (public web API key scraped
  from the folder page + `X-Goog-Drive-Resource-Keys`).
* **Do not run several gdown downloads in parallel.** gdown goes through the
  interactive "can't scan for viruses" page, whose per-IP budget is small: four
  parallel multi-GB downloads got the whole IP throttled, after which even a
  file that had just downloaded fine failed with
  `Cannot retrieve the public link of the file`. `download_peract.py` downloads
  through `files.get?alt=media` with HTTP Range instead - resumable and far
  less trigger-happy. Measured ~4.5 MB/s per stream, so ~77 GB takes a few
  hours at `--jobs 3`.
* Only one of the API keys on the folder page allows both `files.list` and
  `alt=media`; the script probes them and picks the one that works.
