from vllm import LLM, SamplingParams

# 替换成你的模型路径
model_path = "output_models/fedgrpov2_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-12_n500"

# 确保 trust_remote_code=True
llm = LLM(model=model_path, trust_remote_code=True)

# 使用和训练时完全一致的模板
prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$<|im_end|>
<|im_start|>assistant
"""
# 定义停止符
stop_tokens = ["<|im_end|>", "<|endoftext|>", "====="]
num_generated_texts = 4  # 生成的文本数量
sampling_params = SamplingParams(
    temperature=0.6,  # 使用 0 temperature 进行确定性输出
    top_p=0.95,
    max_tokens=2048,
    n=num_generated_texts,
    # repetition_penalty=1.2,
    stop=stop_tokens  # vLLM 的 stop 参数接受字符串列表
)

print('sampling_params:', sampling_params)

outputs = llm.generate([prompt], sampling_params)

if num_generated_texts >1:
    for output in outputs:
        prompt = output.prompt
        print(f"Prompt: {prompt!r}")
        for i, out in enumerate(output.outputs):
            generated_text = out.text
            print(f"Generated text #{i+1}: {generated_text!r}")
        print("-" * 20)
else:
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 20)
