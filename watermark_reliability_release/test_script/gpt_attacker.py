import sys
sys.path.append("/home/shenhm/documents/lm-watermarking/watermark_reliability_release")
from watermark_processor_kwg import WatermarkDetector
import os
import json
from copy import deepcopy
from types import NoneType

from typing import Union
import numpy as np
import sklearn.metrics as metrics
import argparse  
import torch
from utils.submitit import str2bool  # better bool flag type for argparse
from functools import partial
from dataclasses import dataclass
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_new_skipgram_c4/gen_table.jsonl",
        help="Path to the data file containing the z-scores"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default="0.25",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    # parser.add_argument(
    #     "--normalizers",
    #     type=Union[str, NoneType],
    #     default=None,
    #     help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    # )
    parser.add_argument(
        "--ignore_repeated_ngrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    return parser.parse_args()
from openai import OpenAI
import openai
import tenacity
from tenacity import retry, stop_after_attempt, wait_random_exponential
def gpt_attack(example, attack_prompt=None, args=None):
    assert attack_prompt, "Prompt must be provided for GPT attack"
    client = OpenAI(api_key="sk-b45d2bf8d14b43019169bae19a6e9d25", base_url="https://api.deepseek.com")
    gen_row = example

    original_text = gen_row["w_wm_output"]


    attacker_query = attack_prompt + original_text
    query_msg = {"role": "user", "content": attacker_query}

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages, temperature, max_tokens,example):
        try:
            # 发送请求并获取响应
            outputs = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
            attacked_text = outputs.choices[0].message.content
            assert (
                len(outputs.choices) == 1
            ), "OpenAI API returned more than one response, unexpected for length inference of the output"
            example["w_wm_output_attacked_length"] = outputs.usage.completion_tokens
            example["w_wm_output_attacked"] = attacked_text

            print(f"\nOriginal text (T={example['w_wm_output_length']}):\n{original_text}")
            print(f"\nAttacked text (T={example['w_wm_output_attacked_length']}):\n{attacked_text}")
            return example
        
        except openai.BadRequestError as e:
            # 捕获BadRequestError并记录错误，返回一个空文本
            print(f"BadRequestError:  Returning empty text.")
            # 返回一个空文本结果，确保继续处理
            example["w_wm_output_attacked_length"] = 0
            example["w_wm_output_attacked"] = ''
            return example
        except tenacity.RetryError as e:
            # 捕获BadRequestError并记录错误，返回一个空文本
            print(f"BadRequestError:. Returning empty text.")
            # 返回一个空文本结果，确保继续处理
            example["w_wm_output_attacked_length"] = 0
            example["w_wm_output_attacked"] = ''
            return example
        except Exception as e:
            # 捕获BadRequestError并记录错误，返回一个空文本
            print(f"BadRequestError:. Returning empty text.")
            # 返回一个空文本结果，确保继续处理
            example["w_wm_output_attacked_length"] = 0
            example["w_wm_output_attacked"] = ''
            return example
        
    # 调用completion_with_backoff并处理返回值
    example = completion_with_backoff(
        model='gpt-3.5-turbo',
        messages=[query_msg],
        temperature=0.7,
        max_tokens=300,
        example=example
    )

    return example

import json
import os
from tqdm import tqdm


def get_processed_count(output_path):
    """获取已处理的样本数，如果没有记录则返回 0"""
    processed_file = output_path + '.processed'
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
    return 0

def update_processed_count(output_path, count):
    """更新已处理的样本数"""
    processed_file = output_path + '.processed'
    with open(processed_file, 'w', encoding='utf-8') as f:
        f.write(str(count))

def main():
    args = parse_args()

    input_path = args.data_path
    output_path = os.path.splitext(input_path)[0] + '_GPT.jsonl'  # 修改输出路径

    # 获取上次处理到的行数
    processed_count = get_processed_count(output_path)
    prompt = "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'a', encoding='utf-8') as outfile:  # 'a' 模式追加数据
        # 逐行处理，跳过已经处理的行
        for i, line in enumerate(tqdm(infile, desc="Processing samples", initial=processed_count)):
            if i < processed_count:  # 如果当前行已经处理过，则跳过
                continue

            data_item = json.loads(line.strip())  # 读取并解析每一行数据
            if data_item["w_wm_output_length"] < 140:
                print(data_item["w_wm_output_length"],"is too short, pass")
                continue
            updated_item = gpt_attack(data_item, attack_prompt=prompt, args=None)
            outfile.write(json.dumps(updated_item, ensure_ascii=False) + "\n")
            
            # 每处理一个样本，更新已处理行数
            processed_count += 1
            update_processed_count(output_path, processed_count)

    print(f"Updated JSONL file saved to: {output_path}")

if __name__ == '__main__':
    main()
