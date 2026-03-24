# Use NVIDIA PyTorch GPU image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install uv system-wide
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable uv cache and bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy pyproject.toml and lock file
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into the system python environment
RUN uv pip install --system -r pyproject.toml --extra gemma
# Set PYTHONPATH so python knows where to find modules
ENV PYTHONPATH=/app

# Copy only Gemma classification code
COPY src/gemma_classification/ ./gemma_classification/

# Entry point for Vertex AI CustomJob
ENTRYPOINT ["python", "-m", "gemma_classification.finetune.train_gemma_sft"]
