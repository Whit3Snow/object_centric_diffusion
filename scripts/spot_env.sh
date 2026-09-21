#!/bin/bash
# Environment for this SPOT checkout.
#
#   source scripts/spot_env.sh
#
# Works both on the login node and inside an sbatch worker pod: it activates
# the conda env from NAS and points CoppeliaSim / CUDA at paths that exist in
# both places.
#
# NOTE on CUDA: this cluster is driver-only, and /usr/local/cuda-* does not
# exist on worker pods. CUDA_HOME therefore points at the conda env itself
# (cuda-nvcc installed from the nvidia channel), so anything compiled here
# keeps working on a worker.

SPOT_CONDA_ENV="${SPOT_CONDA_ENV:-/home/nas_main/hyojinjang/miniconda3/envs/spot}"
SPOT_CONDA_BASE="${SPOT_CONDA_BASE:-/home/nas_main/hyojinjang/miniconda3}"

if [ -f "$SPOT_CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$SPOT_CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$SPOT_CONDA_ENV"
else
    echo "[spot_env] conda not found at $SPOT_CONDA_BASE" >&2
fi

# ---- conda env's own libs (libGL.so.1 from mesalib, libxkbcommon, ...) ----
# open3d (and therefore the whole SPOT import chain) dlopens libGL.so.1, which
# this cluster does not provide system-wide - only libGLX_nvidia.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# ---- CoppeliaSim / PyRep / RLBench ----
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/nas_main/hyojinjang/CoppeliaSim}"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
# !! Do NOT use QT_QPA_PLATFORM=offscreen here. The offscreen plugin creates no
# OpenGL context, so CoppeliaSim segfaults inside the opengl3 renderer as soon
# as a vision sensor is rendered (QOpenGLFramebufferObject -> shareGroup()).
# Run under Xvfb with the xcb platform instead, and force Mesa's software GL
# (the cluster has no GLX-capable X server on the NVIDIA driver).
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

# ---- CUDA (conda "Path B": self-contained inside the env) ----
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# B200 (sm_100). Set explicitly so source builds target the right arch.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

export HYDRA_FULL_ERROR=1
