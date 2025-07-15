import aiohttp
import asyncio
import logging
from typing import List, Dict, Any

# 设置日志记录，方便调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 响应解析函数 (保持不变) ---
def extract_assistant_response(response_data: Dict[str, Any]) -> List[str] | str:
    """
    从vLLM服务器的响应中提取助手的回答。
    支持单次调用返回单个或多个(n>1)结果的场景。
    """
    if not response_data or 'choices' not in response_data:
        logging.warning("收到的响应格式不正确或为空。")
        return []

    try:
        contents = response_data['choices']
        clean_contents = []
        for content in contents:
            assistant_message = content.get('message', {}).get('content', '')
            # 清理可能存在的特殊字符
            clean_content = assistant_message.replace('\x08', '').strip()
            clean_contents.append(clean_content)

        if len(clean_contents) == 1:
            return clean_contents[0]
        else:
            return clean_contents
    except (KeyError, TypeError, IndexError) as e:
        logging.error(f"解析响应时出错: {e}, 响应数据: {response_data}")
        return []


# --- 优化后的核心查询函数 ---
async def query_vllm_server(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any]
) -> Dict[str, Any] | None:
    """
    使用共享的 aiohttp.ClientSession 向 vLLM 服务器发送单个异步 POST 请求。

    Args:
        session: 复用的 aiohttp.ClientSession 对象。
        url: 目标 vLLM 服务器的 URL。
        payload: 要发送的 JSON 请求体。

    Returns:
        服务器返回的 JSON 字典，如果请求失败则返回 None。
    """
    try:
        async with session.post(url, json=payload) as response:
            # 检查HTTP响应状态码
            if response.status == 200:
                return await response.json()
            else:
                # 如果服务器返回错误，记录状态码和错误信息
                error_text = await response.text()
                logging.error(
                    f"请求失败，URL: {url}, 状态码: {response.status}, 错误信息: {error_text}"
                )
                return None
    except aiohttp.ClientError as e:
        # 处理网络层面的错误，如连接失败
        logging.error(f"请求期间发生客户端错误，URL: {url}, 错误: {e}")
        return None
    except asyncio.TimeoutError:
        # 处理请求超时
        logging.error(f"请求超时，URL: {url}")
        return None


# --- 优化后的主评估函数 ---
async def vllm_evaluate_optimized(
    vllm_urls: List[str],
    batch_prompts: List[str],
    model_names: List[str]
) -> List[str | List[str]]:
    """
    使用复用的 ClientSession 并发地向 vLLM 服务器发送一批请求并获取结果。

    Args:
        vllm_urls: 每个请求对应的 vLLM 服务器 URL 列表。
        batch_prompts: 包含所有提示的列表。
        model_names: 每个请求对应的模型名称列表。

    Returns:
        一个列表，其中每个元素是对应提示的评估结果。
    """
    # 1. 在这里创建 ClientSession，它将在所有请求中被复用
    async with aiohttp.ClientSession() as session:
        tasks = []
        system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses..." # 您可以根据需要自定义

        # 2. 准备所有请求的任务
        for url, prompt, model_name in zip(vllm_urls, batch_prompts, model_names):
            # 将请求体(payload)的构建逻辑放在循环内
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2048,
                "temperature": 1.0,
                "top_p": 0.95,
                # "n": 1, # 如果需要多个返回结果，可以设置 n > 1
            }
            # 将协程任务添加到列表中，注意这里传递了 session 对象
            task = query_vllm_server(session, url, payload)
            tasks.append(task)

        # 3. 并发执行所有任务
        # return_exceptions=True 可以防止一个任务失败导致所有任务被取消
        results = await asyncio.gather(*tasks, return_exceptions=True)

        logging.info(f"从 vLLM 服务器收到了 {len(results)} 个结果。")

        # 4. 处理并解析结果
        eval_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"第 {i} 个请求失败，异常: {result}")
                eval_texts.append(f"ERROR: Request failed - {result}") # 添加错误占位符
            elif result is None:
                logging.warning(f"第 {i} 个请求没有返回有效数据。")
                eval_texts.append("ERROR: No valid data returned") # 添加错误占位符
            else:
                eval_texts.append(extract_assistant_response(result))
    
    logging.info(f"成功解析了 {len(eval_texts)} 个评估文本。")
    return eval_texts

# --- 如何运行的示例 ---
async def main():
    # 假设我们有一个vLLM服务器在本地运行
    # 注意：在实际应用中，URL可能都是同一个
    base_url = "http://localhost:8000/v1/chat/completions"
    
    # 准备一批数据
    prompts = [
        "你好，请介绍一下你自己。",
        "请用Python写一个快速排序算法。",
        "宇宙的起源是什么？",
        "给我讲一个关于人工智能的笑话。",
        # ... 可以添加成百上千个提示
    ]
    num_prompts = len(prompts)
    
    # 假设所有请求都发往同一个URL和使用同一个模型
    urls = [base_url] * num_prompts
    models = ["your-model-name"] * num_prompts # 替换成您的模型名

    print(f"准备发送 {num_prompts} 个请求...")
    
    # 调用优化后的函数
    final_answers = await vllm_evaluate_optimized(urls, prompts, models)
    
    print("\n--- 生成结果 ---")
    for i, answer in enumerate(final_answers):
        print(f"提示 {i+1}: {prompts[i]}")
        print(f"回答: {answer}\n" + "-"*20)

if __name__ == "__main__":
    # 要运行这个异步代码，需要使用 asyncio.run()
    # 请确保您的vLLM服务器正在运行，并替换上面的URL和模型名称
    # asyncio.run(main())
    print("这是一个优化后的代码示例。请取消注释 `asyncio.run(main())` 并配置好URL和模型名称来运行它。")