
import aiohttp
import asyncio

def extract_assistant_response_(input_dict):
    content = input_dict['text'][0]
    
    # 分割获取assistant部分
    # print('content #zgx:', content)
    assistant_section = content.split("assistant\n")[1]
    # assistant_content = assistant_section.split("<|im_end|>")[0]
    
    # 清理特殊字符
    clean_content = assistant_section.replace('\x08', '')
    return clean_content.strip()

    
def extract_assistant_response(input_dict):
    # print('#zgx len(input_dict)', len(input_dict))
    # check the input type
    if isinstance(input_dict, list):
        contents = [input_dict[i]['choices'][0] for i in range(len(input_dict))]
    elif isinstance(input_dict, dict):
        contents = input_dict['choices']
    else:
        raise TypeError("input_dict must be a dict or list")
    # print('len text #zgx:', len(contents))
    clean_contents = []
    for content in contents:
        assistant_section = content['message']['content']
        
        # 清理特殊字符
        clean_content = assistant_section.replace('\x08', '')
        clean_contents.append(clean_content.strip())
    if len(clean_contents) ==1:
        return clean_content.strip()
    else:
        return clean_contents

async def query_model_reward_gen(url, prompt, model_name):
    system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    async with aiohttp.ClientSession() as session:
        # data = {"prompt": prompt, "max_tokens": 1024}
        data = {
            "model": model_name, 
            "messages": [
            {"role": "system", "content": system_prompt}, # 系统消息
            {"role": "user", "content": prompt} # 将 prompt 包装成 messages 列表
            ],
            # "prompt": prompt,  # 结构化消息列表
            "max_tokens": 2048,
            "n": 8,
            "temperature": 0.8,
            "top_p": 0.8, #0.95,
            "repetition_penalty":1.2
        }
        async with session.post(url, json=data) as resp:
            return await resp.json()

async def query_model(url, prompt, model_name):
    system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    async with aiohttp.ClientSession() as session:
        # data = {"prompt": prompt, "max_tokens": 1024}
        data = {
            "model": model_name, 
            "messages": [
            {"role": "system", "content": system_prompt}, # 系统消息
            {"role": "user", "content": prompt} # 将 prompt 包装成 messages 列表
            ],
            # "prompt": prompt,  # 结构化消息列表
            "max_tokens": 2048,
            # "n": 8,
            "temperature": 1.0,
            "top_p": 0.95,
            # "repetition_penalty":1.2
        }
        async with session.post(url, json=data) as resp:
            return await resp.json()

async def vllm_evaluate(vllm_urls, batch_texts, model_names):

    tasks = [ query_model(vllm_url, prompt, model_name) for vllm_url, prompt, model_name in zip(vllm_urls, batch_texts, model_names)
        ]
    results = await asyncio.gather(*tasks)
    print(results[0])
    # print(results[1])
    # print(results[2])
    print('len(vllm_resluts) #zgx:', len(results)) #40
    # print('len(vllm_resluts) #zgx:', len(results['text']))
    eval_texts = []
    if len(results)>1:
        for result in results:
            # import pprint
            # pprint.pprint(results)
            eval_texts.append(extract_assistant_response(result))
    else:
        # import pprint
        # pprint.pprint(results)
        # print('#zgx', results)
        eval_texts = extract_assistant_response(results)
    print(f'len eval_texts : {len(eval_texts)}')
    return eval_texts

async def vllm_evaluate_old(vllm_url, batch_texts, model_name):
    # system_prompt = """As a mathematical problem-solving evaluator, your output MUST strictly follow this structure:

    # ### Analysis Report
    # ```scorecard
    # Evaluate the sloution from the below three dimesions, and the score point output must use double-layer nested tags:
    # <evaluation_steps>
    # 1. Step Completeness: 
    # - [Check solution steps completeness, such as whether contain the answer, and give your detailed analysis here];
    # - <step_completeness><value>0 or 1</value></step_completeness>
    # 2. Step-by-Step Verification: 
    # - [Verify whether the solution is thought step-by-step, and give your detailed analysis here];
    # - <step_by_step><value>0 or 1</value></step_by_step>
    # 3. Answer Correctness: 
    # - [Verify whether the answer is correct through your own computation, and give your detailed analysis here].
    # - <answer_correctness><value>0 or 1</value></answer_correctness>
    # </evaluation_steps>

    # ### Detailed Analysis
    # 1. Step Completeness: 
    # 2. Step-by-Step Verification: 
    # 3. Answer Correctness: 
    # """

    tasks = [ query_model(vllm_url, prompt, model_name) for prompt in batch_texts
        ]
    results = await asyncio.gather(*tasks)
    # print(results[0])
    # print(results[1])
    # print(results[2])
    # print('len(vllm_resluts) #zgx:', len(results))
    # print('len(vllm_resluts) #zgx:', len(results['text']))
    eval_texts = []
    if len(results)>1:
        for result in results:
            # import pprint
            # pprint.pprint(results)
            eval_texts.append(extract_assistant_response(result))
    else:
        # import pprint
        # pprint.pprint(results)
        # print('#zgx', results)
        eval_texts = extract_assistant_response(results)
    print(f'len eval_texts : {len(eval_texts)}')
    return eval_texts

# if __name__ == "__main__":
#     asyncio.run(main())