CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model ./Deepseek-Qwen2.5-7B-Redistil-GRPO \
    --host 0.0.0.0 \
    --port 3347 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32000