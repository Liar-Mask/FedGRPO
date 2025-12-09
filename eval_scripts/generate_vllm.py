#export HF_ENDPOINT=https://hf-mirror.com  
import os
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from math_verify import parse, verify
from oat_math_grader import boxed_reward_fn as oat_evaluate

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def timeout(timeout_seconds: int = 10):
    if os.name == "posix":
        import signal
        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("verify timed out!")
            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrapper
        return decorator

import unicodedata

def normalize_text(text: str) -> str:
    """将字符串中的全角字符（包括数字、字母、标点）转换为半角字符。"""
    return unicodedata.normalize('NFKC', text)

@timeout(timeout_seconds=10)
def labeling_responses(responses: list[str], golden_answer: str):
    # 在解析前，对每个 response 进行规范化处理
    normalized_responses = list(map(normalize_text, responses))
    predict_answers = list(map(parse, normalized_responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_zero_code(question):
    question = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_prime_sft(question, tokenizer):
    # for math problem
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    # for code problem
    # content = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end." 
    msg = [
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def simplerl_template(question: str):
    return (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + question
        + '\nPlease reason step by step, and put your final answer within\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
    )
def fedgrpo_template(question: str):
    return (
        '<|im_start|>system\nYou are a helpful AI Assistant that provides well-reasoned and detailed responses.\nFor a given question, you should first think the reasoning process in the mind step by step and then provide the user with the answer.\nThe reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.\nYou must use the following format:\n<think>\n<put your think process here>\n</think>\n<answer>\n<put your final answer here, do not include anything else except for the final answer>\n</answer>.<|im_end|>\n<|im_start|>user\n'
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=1, force_generate=True, add_think_before_answer=False, add_oat_evaluate=False, any_true=False, skip_scoring=False, output_eval=None, no_split_think=False):
    # n=32
    # force_generate = False
    df = pd.read_parquet(input_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'

    if force_generate or (not os.path.exists(dec_output_path)):
        # 数据处理
        messages = df['prompt'].tolist()
        assert remove_system is True
        if remove_system:
            print('remove system')
            assert messages[0][0]['role'] == 'system'
            messages = [message[1:] for message in messages]
            
        else:
            assert remove_system is False
            print('not remove system')
            
        answers = df['reward_model'].tolist()
        answers = [answer['ground_truth'] for answer in answers]
        # if debug:
            # answers = answers[:10]
        assert len(messages) == len(answers)
                
        print(messages[0])
        print(f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}")
        outputs = generate_vllm(messages, model_path, template=template, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
        # rets = {}
        
        # save the outputs first
        with open(dec_output_path, 'w') as fo:
            for i, output in enumerate(outputs):
                prompt = output.prompt
                for j in range(n):
                    generated_text = output.outputs[j].text
                    item = {
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'answer': answers[i]
                    }
                    fo.write(json.dumps(item) + '\n')
                    
        # format sort prompts, outputs, answers
        assert len(outputs[0].outputs) == n
        prompts = [out.prompt for out in outputs for j in range(n)]
        answers = [answers[i] for i in range(len(outputs)) for j in range(n)]
        outputs = [out.outputs[j].text for out in outputs for j in range(n)]
    else:
        print('Found already decoded file, skip decoding...')
        jss = []
        with open(dec_output_path, 'r') as f:
            for line in f:
                jss.append(json.loads(line))
        
        outputs = [item['generated_text'] for item in jss]
        prompts = [item['prompt'] for item in jss]
        answers = [item['answer'] for item in jss]
    
    data_sources = df['data_source'].tolist()
    
    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg = 0
    from tqdm import tqdm

    print('Scoring...')
    if skip_scoring:
        return
    
    # for i, output in tqdm(enumerate(outputs)):
    diff_cnt = 0
    for i in tqdm(range(len(outputs)), total=len(outputs)):
        # print(i)
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]
        think_format = False
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n') or add_think_before_answer is True:
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            think_format = True
        if no_split_think:
            think_format = False

        labels = None
        if think_format:
            try:
                generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
            except Exception as e:
                labels = [False]
                
        if labels is None:
            try:
                labels = labeling_responses([generated_text,], answer)
            except Exception as e:
                labels = [False]
        
        if add_oat_evaluate:
            new_label = oat_evaluate(generated_text, answer, fast=False)
            new_label = new_label[1] == 1.0
            if any_true is True:
                if labels[0] is False and new_label is True:
                    diff_cnt += 1
                    # breakpoint()
                labels = [labels[0] or new_label]
            else:
                labels = [new_label]

        if n>1:
            rets[data_sources[int((i-i%n)/n)]].append(labels[0])
        else:
            rets[data_sources[i]].append(labels[0])
        
        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0]
        })
        if labels[0]:
            avg += 1

    
    print('accuracy: ', avg / len(outputs))
    print('diff_cnt: ', diff_cnt)
    
    accs = []
    for data_source, labels in rets.items():
        # print(data_source, len(labels))
        acc = np.array(labels).mean()
        print(f'{data_source}: {acc}')
        accs.append(acc)
    
    print('avg acc: ', np.array(accs).mean())
    
    try:
        with open(output_file, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')

def generate_vllm(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=1):
    #vllm模型加载
    # tokenizer_path = 'output_models/grpo/Qwen2.5-Math-7B_data-hendrycks-MATH-benchmark_date-2025-07-05'
    # tokenizer_path = '../simpleR1/outputs/models/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-06-29'
    # tokenizer_path = '../llm_models/Qwen2.5-Math-7B'
    tokenizer_path = model_path
    # tokenizer_path = 'output_models/fedgrpo_2507/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-07-11_0710_n32'
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # max_tokens is for the maximum length for generation.
    

    # 检查并设置 pad_token
    # Qwen系列模型通常没有默认的 pad_token，这在批量推理时可能导致问题。
    # 将其设置为 eos_token 是一个安全且常见的做法。
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     print(f"pad_token 未设置, 已将其设置为 eos_token: {tokenizer.eos_token}")

    # # 获取所有需要停止的 token ID
    # # 对于Qwen Chat模型，<|im_end|> 是一个重要的停止符。
    # # 我们将它和常规的 eos_token 一起作为停止条件。
    # stop_token_ids = []
    
    # # 获取<|im_end|>的ID
    # im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # if isinstance(im_end_token_id, int):
    #     stop_token_ids.append(im_end_token_id)
    #     print(f"添加了停止符 <|im_end|> (ID: {im_end_token_id})")

    # # 获取常规的EOS token ID
    # if tokenizer.eos_token_id is not None:
    #     stop_token_ids.append(tokenizer.eos_token_id)
    #     print(f"添加了停止符 eos_token (ID: {tokenizer.eos_token_id})")
    
    # # 去重，以防 pad_token 和 eos_token 相同
    # stop_token_ids = list(set(stop_token_ids))
    # if not stop_token_ids:
    #     print("警告: 未找到任何有效的 stop_token_id。推理可能不会正常停止。")

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = tokenizer.convert_tokens_to_ids(stop_tokens)
    print(f"Using stop tokens: {stop_tokens} with IDs: {stop_token_ids}")

    # sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=8192, n=n, stop_token_ids=stop_token_ids )
    stop_tokens = ["<|im_end|>", "<|endoftext|>", "====="]
    sampling_params = SamplingParams(temperature=temperature, \
        top_p=top_p, \
        max_tokens=max_tokens, \
        n=n, \
        stop=stop_tokens,\
        # repetition_penalty=1.2 
        )
    print('#debug--sampling_params:', sampling_params)
 
    print(torch.cuda.device_count())
    # llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.85, trust_remote_code=True)  # 替换成本地路径
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.85)  # 替换成本地路径

    gen_prompts = []
    for i in range(len(messages)):
        cur_message = messages[i]
        if template == 'own': 
            gen_prompt = tokenizer.apply_chat_template(
                cur_message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'fedgrpo':
            gen_prompt = fedgrpo_template(cur_message[0]['content'])
        elif template == 'simplerl':
            gen_prompt = simplerl_template(cur_message[0]['content'])
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[0]['content'])
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[0]['content'])
        elif template == 'prime_sft':
            gen_prompt = make_conv_prime_sft(cur_message[0]['content'], tokenizer)
        elif template == 'prime_code':
            gen_prompt = make_conv_zero_code(cur_message[0]['content'])
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else: raise ValueError(f'Invalid template: {template}')
        gen_prompts.append(gen_prompt)
        if i == 0:
            print('Example input: ', gen_prompt)

    outputs = llm.generate(gen_prompts, sampling_params)
    return outputs

if __name__ == "__main__":
    import fire
    fire.Fire(main)
