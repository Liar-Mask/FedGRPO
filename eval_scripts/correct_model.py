import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # 如果您使用了PEFT (LoRA)

# --- 配置 ---
# 原始的基础模型路径 (例如 "Qwen/Qwen2.5-3B-Instruct")
BASE_MODEL_PATH = "output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-09" # ！！！请替换为您的基础模型

# 您想要保存修复后完整模型的路径
SAVE_PATH = "output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-09-corrected" # ！！！请替换为您希望保存的路径

print("正在加载基础模型和 Tokenizer...")
# 从基础模型路径加载，确保配置正确
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto" # 或者 "cpu"
)

# 从您微调的目录加载 Tokenizer，因为它包含了您添加的特殊 Token
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# --- 核心修复步骤 ---
print(f"原始模型 Embedding size: {model.get_input_embeddings().num_embeddings}")
print(f"Tokenizer 词表 size: {len(tokenizer)}")

# 1. 调整模型的 Embedding 层大小以匹配 Tokenizer
model.resize_token_embeddings(len(tokenizer))

print(f"修复后模型 Embedding size: {model.get_input_embeddings().num_embeddings}")


# 2. 保存修复后的、可直接用于 vLLM 的完整模型
print(f"正在将修复后的完整模型保存到: {SAVE_PATH}")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\n修复完成！")
print(f"现在，请在您的 vLLM 推理脚本中将 model_path 指向 '{SAVE_PATH}'。")