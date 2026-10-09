# bowlrl — CPU image by default; GPU (CUDA) variant via --build-arg BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG BASE=python:3.11-slim
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

# EGL/OSMesa for offscreen rendering (videos), ffmpeg for imageio, git for optional MyoSuite assets
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ffmpeg \
        libegl1 libgl1 libglvnd0 libgles2 libosmesa6 libglu1-mesa libglfw3 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python || true

WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY bowlrl ./bowlrl
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data

# CPU torch keeps the image small (~2.5 GB); the CUDA base picks up GPU torch automatically
RUN if [ "${BASE#nvidia}" != "${BASE}" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cu124 ; \
    else \
        pip install torch --index-url https://download.pytorch.org/whl/cpu ; \
    fi \
    && pip install -e ".[dev]"

# runs/ and data/ are mounted from the host so results survive the container
VOLUME ["/app/runs", "/app/data"]
EXPOSE 6006
CMD ["pytest", "-q"]
