# Train via command line


# model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B-Instruct
# model_name_or_path=Qwen/Qwen3-1.7B

# model_name_or_path=meta-llama/Llama-3.2-1B
# model_name_or_path=meta-llama/Llama-3.2-1B-Instruct

# model_name_or_path=Qwen/Qwen2.5-3B
model_name_or_path=../llm_models/Qwen2.5-Math-7B

# CUDA_VISIBLE_DEVICES=4,5 trl vllm-serve \
#     --model ../llm_models/Qwen2.5-Math-7B \
#     --gpu_memory_utilization 0.9 \
#     --tensor_parallel_size 2 \
#     --data_parallel_size 1 

# train_dataset=openai/gsm8k
train_dataset=nlile/hendrycks-MATH-benchmark
# train_dataset=meta-math/MetaMathQA
# train_dataset=SynthLabsAI/Big-Math-RL-Verified
# train_dataset=hiyouga/math12k
# train_dataset=gneubig/aime-1983-2024
# train_dataset=open-r1/OpenR1-Math-220k
# train_dataset=agentica-org/DeepScaleR-Preview-Dataset
# train_dataset=RUC-AIBOX/STILL-3-Preview-RL-Data
# train_dataset=Maxwell-Jia/AIME_2024



eval_dataset=HuggingFaceH4/MATH-500
# eval_dataset=opencompass/AIME2025


model_name=$(basename $model_name_or_path)
# run_name=$model_name-$(date +%Y-%m-%d)
run_name=${model_name}_data-$(basename $train_dataset)_date-$(date +%Y-%m-%d)-lora


OUTPUT_DIR=output_models/grpo/$run_name
LOG_FILE="$OUTPUT_DIR/train_log_$(date +%Y-%m-%d_%H:%M:%S.log)"

mkdir -p $OUTPUT_DIR

echo "current file is: $0"
cp "$0" "$OUTPUT_DIR"/run.sh


echo
echo "==================== Training: ===================="
echo "[INFO] run name: $run_name"
echo "[INFO] logs are saved to: $LOG_FILE"
echo

# sleep 7h


MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=../.cache/huggingface
# export HF_HOME=/xxx/.cache/huggingface
# Set `HF_HOME` when necessary, by default HF_HOME=~/.cache/huggingface
# where `~` is user's home directory.

accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=4 \
GRPO.py \
    --config recipes/Qwen2.5-Math-7B/config_demo_mathlight.yaml \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $model_name_or_path \
    --dataset_name $train_dataset \
    --num_train_epochs 1 \
    --num_generations 5 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 5 \
    --num_iterations 3 \
    --torch_empty_cache_steps 1 \
    --max_num_train_samples 2000 \
    --max_completion_length 4096 \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_mode server \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port 8000 \
    --reward_funcs accuracy format tag_count \
    --reward_weights 8 1 1 \
    --loss_type bnpo \
    --scale_rewards False \
    --mask_truncated_completions True \
    --epsilon 0.2 \
    --epsilon_high 0.3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --beta 0.01 \
    --lr_scheduler_type constant \
    --learning_rate 3e-6 \
    --save_strategy epoch \
    --save_steps 400 \
    --eval_on_start False \
    --log_level info \
    --wandb_project fedgrpo-$(basename $train_dataset) \
    --run_name $run_name \
    2>&1 | tee $LOG_FILE





