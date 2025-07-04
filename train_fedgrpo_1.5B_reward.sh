# Train via command line



model_name_or_path=Qwen/Qwen2.5-1.5B
# model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B-Instruct
# model_name_or_path=Qwen/Qwen3-1.7B

# model_name_or_path=meta-llama/Llama-3.2-1B
# model_name_or_path=meta-llama/Llama-3.2-1B-Instruct

# model_name_or_path=Qwen/Qwen2.5-3B
model_name_or_path=../llm_models/Qwen2.5-Math-1.5B

# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model \
#  $model_name_or_path --gpu_memory_utilization 0.85 


# train_dataset=openai/gsm8k
train_dataset=nlile/hendrycks-MATH-benchmark
train_dataset='../llm_datasets/MATH-benchmark-reward-grpo/train'
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

train_mode='reward_train'
problem_types='Intermediate_Algebra Prealgebra'
problem_types='Number_Theory Geometry'
problem_types='Precalculus Counting_&_Probability'
client_id=0

model_name=$(basename $model_name_or_path)
# run_name=$model_name-$(date +%Y-%m-%d)
# run_name=${model_name}_data-$(basename $train_dataset)_date-$(date +%Y-%m-%d)
run_name=${model_name}_data-MATH-benchmark-${problem_types// /_}_client${client_id}_date-$(date +%Y-%m-%d)


OUTPUT_DIR=output_models/grpo_reward/$run_name
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

export CUDA_VISIBLE_DEVICES=4
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
    --num_processes=1 \
GRPO.py \
    --config recipes/Qwen2.5-Math-1.5B/config_demo_mathlight.yaml \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $model_name_or_path \
    --dataset_name $train_dataset \
    --train_mode $train_mode \
    --problem_types $problem_types \
    --client_id $client_id \
    --num_train_epochs 1 \
    --num_generations 6 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 3 \
    --num_iterations 3 \
    --torch_empty_cache_steps 1 \
    --max_completion_length 2048 \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.25 \
    --vllm_mode colocate \
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
    --beta 0.0001 \
    --lr_scheduler_type constant \
    --learning_rate 3e-6 \
    --save_strategy steps \
    --save_steps 100 \
    --log_level info \
    --wandb_project grpo-$(basename $train_dataset) \
    --run_name $run_name \
    2>&1 | tee $LOG_FILE





