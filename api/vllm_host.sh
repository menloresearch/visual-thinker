vllm serve jan-hq/AlphaMaze-v0.2-1.5B-GRPO-cp-1600 \
    --served-model-name jan-hq/AlphaMaze-v0.2-1.5B-GRPO-cp-600 \
    --host 0.0.0.0 \
    --port 3347 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32000
