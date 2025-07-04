
## 先运行vllm-server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model \
../llm_models/Qwen2.5-Math-1.5B-Instruct --gpu_memory_utilization 0.85  

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model \
../llm_models/Qwen2.5-3B-Instruct --gpu_memory_utilization 0.85 

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model \
output_models/fedgrpo/Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0620 --gpu_memory_utilization 0.85  

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model \
../llm_models/Qwen2.5-Math-7B --gpu_memory_utilization 0.85  


## 执行FedGRPO
log_dir='logs/fedgrpo-2506'
mkdir -p $log_dir
# Qwen2.5-Math-1.5B-Instruct
CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 FedGRPO.py \
    --config recipes/Qwen2.5-Math-1.5B/config_demo_mathlight.yaml \
    >> $log_dir/Qwen2.5-Math-1.5B-mathlight-fedgrpo-0616v4.log 2>&1 
# Qwen2.5-3B-Instruct
model_name='Qwen2.5-Math-7B'
model_name=Qwen2.5-Math-1.5B
model_name=Qwen2.5-3B-Instruct
CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 FedGRPO.py \
    --config recipes/${model_name}/config_demo_mathlight.yaml \
    >> $log_dir/${model_name}-mathlight-fedgrpo-lenrward-4096-0621.log 2>&1 

CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=1 FedGRPO.py \
    --config recipes/${model_name}/config_demo_mathlight.yaml \
    >> $log_dir/${model_name}-mathlight-fedgrpo-lenrward-2048-0623.log 2>&1 

CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=1 FedGRPO.py \
    --config recipes/${model_name}/config_demo_mathlight.yaml \
    >> $log_dir/${model_name}-mathlight-fedgrpo-lenrward-2048-0624-lr5e6.log 2>&1 


## 测试fedgrpo

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 1.zero-shot
MODEL=../llm_models/Qwen2.5-Math-1.5B-Instruct
OUTPUT_DIR=logs/evals/zeroshot/Qwen2.5-Math-1.5B-Instruct-zero

#2.FedGRPO fedgrpo/output_models/fedgrpo/Qwen2.5-Math-1.5B-mathlight-fedgrpo
MODEL='output_models/fedgrpo/Qwen2.5-Math-1.5B-mathlight-fedgrpo'
OUTPUT_DIR=logs/evals/fedgrpo/Qwen2.5-Math-1.5B-mathlight-fedgrpo

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"


TASK=math_500
# TASK=aime24
#TASK=gpqa:diamond
#TASK=aime25
CUDA_VISIBLE_DEVICES=0 lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


## merge 7B model lora adapters

python merge_lora.py \
  --base_model ../llm_models/Qwen2.5-Math-7B-Instruct \
  --lora_dir ./output_models/fedgrpo/Qwen2.5-Math-7B-mathlight-fedgrpo \
  --output_dir ./output_models/fedgrpo/lora_merged_models/Qwen2.5-Math-7B-mathlight-fedgrpo





