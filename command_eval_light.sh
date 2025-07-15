export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT=https://hf-mirror.com

MODEL=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10

OUTPUT_DIR=logs/evals/grpo/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"


TASK=math_500
TASK=aime24
#TASK=gpqa:diamond
#TASK=aime25
#TASK=livemathbench
CUDA_VISIBLE_DEVICES=0 lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR