
## eval form LUFFY (https://github.com/ElliottYan/LUFFY)

ROOT=$(pwd) #YOUR_ROOT_PATH
DATA=$ROOT/exp_dataset/valid.all.parquet

OUTPUT_DIR=logs/evals/fedgrpo #./results/

OUTPUT_DIR=logs/evals/fedgrpo_w_reward
OUTPUT_DIR=logs/evals/fedgrpov2_2507
# OUTPUT_DIR=logs/evals/grpo
# OUTPUT_DIR=logs/evals/fedgrpo_2507
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=../llm_models/Qwen2.5-Math-1.5B-Instruct
MODEL_NAME=Qwen2.5-Math-1.5B-Instruct-zero

MODEL_PATH=output_models/fedgrpo/Qwen2.5-Math-1.5B-mathlight-fedgrpo
MODEL_NAME=Qwen2.5-Math-1.5B-mathlight-fedgrpo-n32

MODEL_PATH=output_models/fedgrpo/lora_merged_models/Qwen2.5-Math-7B-mathlight-fedgrpo
MODEL_NAME=Qwen2.5-Math-7B-Instruct-mathlight-fedgrpo

# OUTPUT_DIR=logs/evals/zeroshot
MODEL_PATH=../llm_models/Qwen2.5-Math-7B-Instruct
MODEL_NAME=Qwen2.5-Math-7B-Instruct-zero

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

# MODEL_PATH=output_models/grpo/Qwen2.5-Math-7B_data-hendrycks-MATH-benchmark_date-2025-07-05
# MODEL_NAME=Qwen2.5-Math-7B_data-hendrycks-MATH-benchmark_date-2025-07-05

# MODEL_PATH=output_models/grpo/Qwen2.5-Math-7B_data-Openr1-Math-46k-orginal_date-2025-07-06
# MODEL_NAME=Qwen2.5-Math-7B_data-Openr1-Math-46k-orginal_date-2025-07-06

# MODEL_PATH=output_models/fedgrpo/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-07
# MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-07

# MODEL_PATH=output_models/fedgrpo/Qwen2.5-3B_data-Openr1-Math-46k-orginal_date-2025-07-08
# MODEL_NAME=Qwen2.5-3B_data-Openr1-Math-46k-orginal_date-2025-07-08

MODEL_PATH=../simpleR1/outputs/models/fedgrpo/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-01
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-01-simpleR1

MODEL_PATH=output_models/fedgrpo_w_reward/Qwen2.5-3B_data-train_date-2025-07-09
MODEL_NAME=Qwen2.5-3B_data-MATH-benchmark-train_date-2025-07-09

MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-09
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-09

MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-09-corrected
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-09-corrected


MODEL_PATH=output_models/grpo/Qwen2.5-Math-7B_data-Openr1-Math-46k-orginal_date-2025-07-06
MODEL_NAME=Qwen2.5-Math-7B_data-Openr1-Math-46k-orginal_date-2025-07-06

MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10

MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10_0710test
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10_0710test

MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10_0710_n1000
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-10_0710_n1000

# MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11_0710_n32
# MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11_0710_n32

MODEL_PATH=output_models/grpo/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11

MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11_n1000
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11_n1000

# MODEL_PATH=output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11_n500_w.o.fed
# MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11_n500_w.o.fed

MODEL_PATH=output_models/fedgrpov2_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-12_n500
MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-12_n500

MODEL_PATH=../llm_models/Qwen2.5-3B
MODEL_NAME=Qwen2.5-3B-zero-0710

# MODEL_PATH=output_models/fedgrpov2_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-14_n500
# MODEL_NAME=Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-14_n500

# MODEL_PATH=output_models/fedgrpo/Qwen2.5-Math-7B_data-hendrycks-MATH-benchmark_date-2025-07-13
# MODEL_NAME=Qwen2.5-Math-7B_data-hendrycks-MATH-benchmark_date-2025-07-13

# MODEL_PATH=output_models/grpo/Qwen2.5-Math-7B_data-Openr1-Math-46k-orginal_date-2025-07-06
# MODEL_NAME=Qwen2.5-Math-7B_data-Openr1-Math-46k-orginal_date-2025-07-06

MODEL_PATH=output_models/fedgrpov2_2507/Qwen2.5-Math-1.5B_data-hendrycks-MATH-benchmark_date-2025-07-14_beta0.01
MODEL_NAME=Qwen2.5-Math-1.5B_data-hendrycks-MATH-benchmark_date-2025-07-14_beta0.01

# MODEL_PATH=output_models/fedgrpov2_2507/Qwen2.5-Math-1.5B_data-hendrycks-MATH-benchmark_date-2025-07-14
# MODEL_NAME=Qwen2.5-Math-1.5B_data-hendrycks-MATH-benchmark_date-2025-07-14

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi
# TEMPLATE=qwen
# TEMPLATE=own
TEMPLATE=fedgrpo
# TEMPLATE=simplerl
TEMPLATE=qwen
echo "Using model: $MODEL_NAME"
echo "Using template: $TEMPLATE"
echo "Using data: $DATA"
echo 'output file: '$OUTPUT_DIR/${MODEL_NAME}_${TEMPLATE}_2048_eos.jsonl

CUDA_VISIBLE_DEVICES=3 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --max_tokens 2048 \
  --remove_system True \
  --add_oat_evaluate True \
  --force_generate False \
  --output_file $OUTPUT_DIR/${MODEL_NAME}_${TEMPLATE}_2048_eos.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/${MODEL_NAME}_${TEMPLATE}_2048_eos.log


