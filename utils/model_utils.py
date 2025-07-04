from transformers import AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig

from configs import GRPOConfig, SFTConfig


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer

import re
from bs4 import BeautifulSoup

def extract_scores(texts):
    # 如果不是list;  如果是list for循环 rewards = []
    if not isinstance(texts, list):
        texts = [texts]
    rewards = []
    for text in texts:
        step_completeness_score, step_by_step_score, answer_correctness_score, total_score =None, None, None,None
        try:
            scorecard = re.search(r'```scorecard(.*?)```', text, re.DOTALL).group(1)
            soup = BeautifulSoup(scorecard, 'xml')  # 使用更健壮的解析器
            
            # 添加容错判断
            step_completeness = soup.find('step_completeness').find('value')
            step_completeness_score = int(step_completeness.text) if step_completeness else 0

            step_by_step = soup.find('step_by_step').find('value')
            step_by_step_score = int(step_completeness.text) if step_by_step else 0

            answer_correctness = soup.find('answer_correctness').find('value')
            answer_correctness_score = int(answer_correctness.text) if answer_correctness else 0
            
            # total_score = soup.find('total_score').find('value')
            # total = int(total_score.text) if total_score else 0

            total_score= step_completeness_score+step_by_step_score+answer_correctness_score
            # return {
            #     'step_completeness': step_completeness_score,
            #     'step_by_step': step_by_step_score,
            #     'answer_correctness': answer_correctness_score,
            #     'total_score': total_score
            # }
        except Exception as e:
            print(f"解析失败：{str(e)}")
            # return {
            #     'step_completeness': step_completeness_score,
            #     'step_by_step': step_by_step_score,
            #     'answer_correctness': answer_correctness_score,
            #     'total_score': total_score
            # }
        rewards.append(total_score)
    return rewards



def extract_scores_(texts):
    """
    从分析报告中提取三个评估数值的函数
    参数:
        text (str): 包含XML格式评分标签的文本
    返回:
        dict: 包含三个键值对的字典，若未找到则值为None
    """
    patterns = {
        'step_completeness': r'<step_completeness><value>(\d+)</value></step_completeness>',
        'step_by_step': r'<step_by_step><value>(\d+)</value></step_by_step>',
        'answer_correctness': r'<answer_correctness><value>(\d+)</value></answer_correctness>'
    }
    if not isinstance(texts, list):
        texts = [texts]
    rewards = []
    for text in texts:
        result = 0
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            result += int(match.group(1)) if match else 0
        rewards.append(result)
    return rewards


def extract_scores(texts):
    """
    从分析报告中提取三个评估数值的函数
    参数:
        texts (str/list): 包含多种格式评分标识的文本或列表
    返回:
        list: 包含各文本总分的列表，未找到则计0分
    """
    patterns = {
        'step_completeness': (
            r'(?:<step_completeness>(?:<value>)?(\d+)(?:</value>)?</step_completeness>'  # XML标签格式
            r'|#### Step Completeness:[\s\S]*?Score[^\d]*(\d+))'                        # Markdown评分格式
        ),
        'step_by_step': (
            r'(?:<step_by_step>(?:<value>)?(\d+)(?:</value>)?</step_by_step>'
            r'|#### Step-by-Step Verification:[\s\S]*?Score[^\d]*(\d+))'
        ),
        'answer_correctness': (
            r'(?:<answer_correctness>(?:<value>)?(\d+)(?:</value>)?</answer_correctness>'
            r'|#### Answer Correctness:[\s\S]*?Score[^\d]*(\d+))'
        )
    }
    
    if not isinstance(texts, list):
        texts = [texts]
    
    rewards = []
    for text in texts:
        total = 0
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                # 提取第一个非空的捕获组
                groups = match.groups()
                value = next((float(g) for g in groups if g is not None), 0)
                if value > 1.0:
                    value = 1.0
                total += value
        rewards.append(total)
        if total == 0: 
            print('extract 0 score text#zgx:', text)
    return rewards

        
