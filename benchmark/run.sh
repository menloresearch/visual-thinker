export CUDA_VISIBLE_DEVICES=1
# python main.py \
#     --engine-type hf \
#     --model-name "jan-hq/AlphaMaze-v0.1-1.5B-GRPO-cp-600" \
#     --instruction-type "alphamaze" \
#     --temperature 0.6 \
#     --max-tokens 20000 \
#     --batch-size 4
python main.py \
    --engine-type vllm \
    --model-name "jan-hq/AlphaMaze-v0.1-1.5B-GRPO-cp-600" \
    --instruction-type "alphamaze" \
    --temperature 0.6 \
    --max-tokens 20000 \
    --batch-size 8