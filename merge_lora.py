"""
python merge_lora.py \
  --base_model ./models/llama-2-7b-hf \
  --lora_dir ./lora_adapters/my_lora \
  --output_dir ./merged_models/llama-2-7b-lora
"""

import argparse
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def copy_config_files(src_path, dst_path):
    """复制原始模型的配置文件到目标目录"""
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源目录不存在: {src_path}")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    # 复制所有非权重文件（.bin/.safetensors）
    exclude_ext = {'.bin', '.safetensors', '.pt'}
    for item in os.listdir(src_path):
        src_item = os.path.join(src_path, item)
        if os.path.splitext(item)[1] not in exclude_ext:
            dst_item = os.path.join(dst_path, item)
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dst_item)

def merge_lora_adapter(base_model_path, lora_path, output_path):
    """合并基础模型和LoRA适配器"""
    # 加载基础模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    lora_model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto")
    
    # 合并权重
    merged_model = lora_model.merge_and_unload()
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存合并模型
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # 复制配置文件
    copy_config_files(base_model_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并基础模型与LoRA适配器')
    parser.add_argument('--base_model', type=str, required=True, help='基础模型目录路径')
    parser.add_argument('--lora_dir', type=str, required=True, help='LoRA适配器目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='合并输出目录路径')
    
    args = parser.parse_args()
    
    # 执行合并操作
    merge_lora_adapter(
        base_model_path=args.base_model,
        lora_path=args.lora_dir,
        output_path=args.output_dir
    )
    
    print(f"模型已成功合并并保存至: {args.output_dir}")