# Parakeet RNNT Hindi Fine-Tuning Docker Container
# Based on NVIDIA NeMo container for maximum compatibility
#
# Build: docker build -t parakeet-finetune .
# Run:   docker run --gpus all -v $(pwd):/workspace parakeet-finetune

FROM nvcr.io/nvidia/nemo:24.07

# Set working directory
WORKDIR /workspace

# Install additional dependencies for HuggingFace integration
RUN pip install --no-cache-dir \
    datasets>=2.14.0 \
    huggingface_hub>=0.20.0 \
    jiwer>=3.0.0 \
    pandas>=2.0.0

# Create directory structure
RUN mkdir -p /workspace/data /workspace/outputs /workspace/tokenizers /workspace/configs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV NEMO_CACHE_DIR=/workspace/.nemo_cache

# Default command
CMD ["/bin/bash"]
