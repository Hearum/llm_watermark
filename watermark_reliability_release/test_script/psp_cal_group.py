
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
from evaluate import load
from argparse import Namespace
import os
import subprocess

from sacremoses import MosesTokenizer
from metrics.p_sp_utils.models import load_model
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from metrics.p_sp_utils.data_utils import get_df
from metrics.p_sp_utils.evaluate_sts import Example

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_4_32_0.2_LSH_v2.2_c4_new/gen_table_meta.json",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        help="The seeding procedure to use for the watermark.",
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

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class FileSim(object):

    def __init__(self):
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def score(self, params, batcher, input1, input2, use_sent_transformers=False):
        sys_scores = []
        if not use_sent_transformers:
            for ii in range(0, len(input1), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            #Compute embedding for both lists
            for i in range(len(input1)):
                embedding_1= model.encode(input1[i], convert_to_tensor=True)
                embedding_2 = model.encode(input2[i], convert_to_tensor=True)

                score = util.pytorch_cos_sim(embedding_1, embedding_2)
                sys_scores.append(score.item())
        return sys_scores

def batcher(params, batch):
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        if params.lower_case:
            p = p.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        p = Example(p, params.lower_case)
        p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
        new_batch.append(p)
    x, l = params.model.torchify_batch(new_batch)
    vecs = params.model.encode(x, l)
    return vecs.detach().cpu().numpy()

def evaluate_p_sp(input1, input2, use_sent_transformers=False):
    download_url = 'http://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip'
    download_dir = './metrics/p_sp_utils'

    args = {
        'gpu': 1 if torch.cuda.is_available() else 0,
        'load_file': '/home/shenhm/documents/lm-watermarking/watermark_reliability_release/metrics/p_sp_utils/paraphrase-at-scale-english/model.para.lc.100.pt',
        'sp_model': '/home/shenhm/documents/lm-watermarking/watermark_reliability_release/metrics/p_sp_utils/paraphrase-at-scale-english/paranmt.model',
    }

    # Check if the required files exist
    if not os.path.exists(args['load_file']) or not os.path.exists(args['sp_model']):
        # make a box around the print statement
        print("====================================="*2)
        print("Pretrained model weights wasn't found, Downloading paraphrase-at-scale-english.zip...")
        print("====================================="*2)
        # Download the zip file
        subprocess.run(['wget', download_url])

        # Unzip the file
        subprocess.run(['unzip', 'paraphrase-at-scale-english.zip', '-d', download_dir])

        # Delete the zip file
        os.remove('paraphrase-at-scale-english.zip')

        # Update the file paths
        args['load_file'] = os.path.join(download_dir, 'paraphrase-at-scale-english/model.para.lc.100.pt')
        args['sp_model'] = os.path.join(download_dir, 'paraphrase-at-scale-english/paranmt.model')

    model, _ = load_model(None, args)
    model.eval()

    entok = MosesTokenizer(lang='en')

    new_args = Namespace(batch_size=32, entok=entok, sp=model.sp,
                     params=args, model=model, lower_case=model.args.lower_case,
                     tokenize=model.args.tokenize)
    s = FileSim()
    scores = s.score(new_args, batcher, input1, input2, use_sent_transformers)
    return scores

from tqdm import tqdm
import math
def convert_item(item):
    for key, value in item.items():
        if isinstance(value, np.float32):  # 检查是否为 float32 类型
            item[key] = float(value)  # 转换为 Python 原生的 float
    return item

def read_data_from_file(data_path):
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))  # 读取每一行并转换为字典
        print(f"Data loaded successfully from {data_path}")
    except Exception as e:
        print(f"Error while loading data from {data_path}: {e}")
    return data

def main(data_path):
    
    args = parse_args()
    data = []
    output_file = data_path + '_psp'
    if os.path.exists(output_file):
        print('#' * 50, "result", '#' * 50)
        data = read_data_from_file(output_file)
        print('#' * 50, "result", '#' * 50)
        psp_values = [item.get('wm_and_no_wm_psp', float('nan')) for item in data if not math.isnan(item.get('wm_and_no_wm_psp', float('nan')))]
        if psp_values:
            mean_psp = np.mean(psp_values)
        else:
            mean_psp = math.nan
        print("Mean wm_and_no_wm_psp:", mean_psp)
        return
    

    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # 不再需要perplexity模块，改为调用evaluate_p_sp来计算P-SP
    tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9", local_files_only=True)

    batch_size = 32  # 可以根据实际内存和计算能力调整批次大小
    batch_data = []  # 用于存储当前批次的文本
    batch_keys = []  # 用于存储对应文本的key（用于后续赋值）

    for item in tqdm(data):
        # 获取 w_wm_output 和 no_wm_output
        w_wm_output = item.get("w_wm_output", "")
        no_wm_output = item.get("no_wm_output", "")

        # 如果 w_wm_output 和 no_wm_output 均为字符串，则加入到批次中
        if isinstance(w_wm_output, str) and isinstance(no_wm_output, str):
            # 对输入文本进行编码并限制最大长度为160
            tokens_w_wm = tokenizer.encode(w_wm_output, truncation=True, max_length=160)
            tokens_no_wm = tokenizer.encode(no_wm_output, truncation=True, max_length=160)

            # 对于有效的文本才进行处理
            if len(tokens_w_wm) > 10 and len(tokens_no_wm) > 10:
                batch_data.append(w_wm_output)
                batch_data.append(no_wm_output)
                batch_keys.append((item, "w_wm_output", "no_wm_output"))

        # 如果当前批次已经达到设定的大小，则计算P-SP
        if len(batch_data) >= batch_size:
            try:
                # 使用evaluate_p_sp来计算P-SP，相似度得分
                results = evaluate_p_sp(batch_data[::2], batch_data[1::2])  # 取出交替的文本对
                for idx, (item, key1, key2) in enumerate(batch_keys):
                    item['wm_and_no_wm_psp'] = float(round(results[idx], 2))  # 保存P-SP相似度分数
            except (ValueError, AssertionError) as e:
                if "Must have at least 1 token" in str(e):
                    for idx, (item, key1, key2) in enumerate(batch_keys):
                        item['wm_and_no_wm_psp'] = math.nan
                else:
                    raise
            batch_data = []
            batch_keys = []

    # 处理剩余的文本（如果有）
    if batch_data:
        try:
            results = evaluate_p_sp(batch_data[::2], batch_data[1::2])  # 取出交替的文本对
            for idx, (item, key1, key2) in enumerate(batch_keys):
                item['wm_and_no_wm_psp'] = round(results[idx], 2)
        except ValueError as e:
            if "Must have at least 1 token" in str(e):
                for idx, (item, key1, key2) in enumerate(batch_keys):
                    item['wm_and_no_wm_psp'] = math.nan
            else:
                raise

    # 写回 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            item = convert_item(item)
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Updated JSONL file saved to: {output_file }")

    # 打印结果
    print('#' * 50, "result", '#' * 50)
    psp_values = [item.get('wm_and_no_wm_psp', float('nan')) for item in data if not math.isnan(item.get('wm_and_no_wm_psp', float('nan')))]
    if psp_values:
        mean_psp = np.mean(psp_values)
    else:
        mean_psp = math.nan
    print("Mean wm_and_no_wm_psp:", mean_psp)

if __name__ == '__main__':
    paths = [
            "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
            "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf16-4-True-15485863/gen_table.jsonl",
            "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf32-4-True-15485863/gen_table.jsonl",
            "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf128-4-True-15485863/gen_table.jsonl",
             "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf1-4-True-15485863/gen_table.jsonl",
        ]

    for path in paths:
        main(path)