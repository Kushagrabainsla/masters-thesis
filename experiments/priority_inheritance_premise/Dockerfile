FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_TARGET_DEVICE=empty

ARG VLLM_SHA=3897f95cf17619137536d2a7d325d9c2f39dc38f

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /opt/vllm \
    && git -C /opt/vllm checkout --detach "$VLLM_SHA"

# CPU PyTorch at the version that the pinned vLLM requires for aarch64.
RUN pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r /opt/vllm/requirements/common.txt

# The [build-system] requirements of the pinned pyproject.toml (torch is above).
RUN pip install "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" \
    "packaging>=24.2" wheel jinja2 ninja "cmake>=3.26.1" pytest

# Python-only vLLM: VLLM_TARGET_DEVICE=empty builds no kernels. The Rust extensions are
# optional by default (VLLM_REQUIRE_RUST_FRONTEND is unset), so no Rust toolchain is needed.
RUN pip install --no-build-isolation --no-deps -e /opt/vllm

WORKDIR /experiment
ENTRYPOINT ["python3"]
