# bowlrl — CPU image by default; GPU (CUDA) variant via --build-arg BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG BASE=python:3.11-slim
FROM ${BASE}
ARG BASE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

# EGL/OSMesa for offscreen rendering (videos), ffmpeg for imageio, git for optional MyoSuite assets.
# The python:* base already ships Python; the CUDA base needs it from apt.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libegl1 libgl1 libglvnd0 libgles2 libosmesa6 libglu1-mesa libglfw3 \
    && if ! command -v python3 >/dev/null 2>&1; then \
         apt-get install -y --no-install-recommends python3 python3-pip python3-venv; fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY bowlrl ./bowlrl
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data

# CPU torch keeps the image small; the CUDA base gets GPU torch
RUN python3 -m pip install --upgrade pip \
    && case "${BASE}" in \
         nvidia*) python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124 ;; \
         *)       python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu ;; \
       esac \
    && python3 -m pip install -e ".[dev]"

# runs/ and data/ are mounted from the host so results survive the container
VOLUME ["/app/runs", "/app/data"]
EXPOSE 6006
CMD ["python3", "-m", "pytest", "-q"]
