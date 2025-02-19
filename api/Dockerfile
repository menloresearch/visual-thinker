FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /app

# Install vLLM
RUN pip install --upgrade pip && \
    pip install vllm==0.3.3 --no-cache-dir

# Copy required files
COPY config.yml api_tokens.yml ./
COPY vllm_host.sh ./

# Make script executable
RUN chmod +x vllm_host.sh

# Expose the port
EXPOSE 3347

# Run vLLM server
CMD ["/bin/bash", "./vllm_host.sh"]