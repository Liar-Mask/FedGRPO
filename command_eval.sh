
## eval form LUFFY (https://github.com/ElliottYan/LUFFY)

ROOT=$(pwd) #YOUR_ROOT_PATH
DATA=$ROOT/exp_dataset/valid.all.parquet

OUTPUT_DIR=logs/evals/fedgrpo #./results/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=../llm_models/Qwen2.5-Math-1.5B-Instruct
MODEL_NAME=Qwen2.5-Math-1.5B-Instruct-zero

MODEL_PATH=output_models/fedgrpo/Qwen2.5-Math-1.5B-mathlight-fedgrpo
MODEL_NAME=Qwen2.5-Math-1.5B-mathlight-fedgrpo-n32

MODEL_PATH=output_models/fedgrpo/lora_merged_models/Qwen2.5-Math-7B-mathlight-fedgrpo
MODEL_NAME=Qwen2.5-Math-7B-Instruct-mathlight-fedgrpo

OUTPUT_DIR=logs/evals/zeroshot
MODEL_PATH=../llm_models/Qwen2.5-Math-7B-Instruct
MODEL_NAME=Qwen2.5-Math-7B-Instruct-zero

OUTPUT_DIR=logs/evals/fedgrpo
MODEL_PATH=output_models/fedgrpo/Qwen2.5-Math-1.5B-mathlight-fedgrpo-len-0620-1024
MODEL_NAME=Qwen2.5-Math-1.5B-mathlight-fedgrpo-len-0620-1024-t2

MODEL_PATH=output_models/fedgrpo/Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0620
MODEL_NAME=Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0622

MODEL_PATH=output_models/fedgrpo/Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0623-2048
MODEL_NAME=Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0623-2048-ep2-qwentmp

MODEL_PATH=output_models/fedgrpo/Qwen2.5-Math-1.5B-mathlight-fedgrpo-len-0620-2048
MODEL_NAME=Qwen2.5-Math-1.5B-mathlight-fedgrpo-len-0620-2048-qwentmp

MODEL_PATH=../llm_models/Qwen2.5-Math-1.5B-Instruct
MODEL_NAME=Qwen2.5-Math-1.5B-Instruct-zero-qwentmp

MODEL_PATH=output_models/fedgrpo/Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0624-2048-lr5e6
MODEL_NAME=Qwen2.5-3B-Instruct-mathlight-fedgrpo-len-0624-2048-lr5e6

MODEL_PATH=output_models/Qwen2.5-Math-1.5B_data-hendrycks-MATH-benchmark_date-2025-07-02
MODEL_NAME=Qwen2.5-Math-1.5B_data-hendrycks-MATH-benchmark_date-2025-07-02

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=1 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log


