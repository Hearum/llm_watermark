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
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new/gen_table.jsonl",
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
    parser.add_argument(
        "--l",
        type=int,
        default=60,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--o",
        type=int,
        default=60,
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
import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# nltk.download("punkt")

def dipper_attacker(
    dd,
    model,
    tokenizer,
    no_ctx=True,
    sent_interval=3,
    start_idx=None,
    end_idx=None,
    paraphrase_file=".output/dipper_attacks.jsonl",
    lex=0,
    order=0,
    args=None,
):

    if "w_wm_output_attacked" not in dd:
        # paraphrase_outputs = {}

        if isinstance(dd["w_wm_output"], str):
            input_gen = dd["w_wm_output"].strip()
        else:
            input_gen = dd["w_wm_output"][0].strip()

        # The lexical and order diversity codes used by the actual model correspond to "similarity" rather than "diversity".
        # Thus, for a diversity measure of X, we need to use control code value of 100 - X.
        lex_code = int(100 - lex)
        order_code = int(100 - order)

        # remove spurious newlines
        input_gen = " ".join(input_gen.split())
        sentences = sent_tokenize(input_gen)
        prefix = " ".join(dd["truncated_input"].replace("\n", " ").split())
        output_text = ""
        final_input_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            if no_ctx:
                final_input_text = f"lexical = {lex_code}, order = {order_code} <sent> {curr_sent_window} </sent>"
            else:
                final_input_text = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {curr_sent_window} </sent>"

            if lex_code == 60 and order_code == 60:
                print(final_input_text)

            final_input = tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = model.generate(
                    **final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512
                )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        # paraphrase_outputs[f"lex_{lex_code}_order_{order_code}"] = {
        #     "final_input": final_input_text,
        #     "output": [output_text.strip()],
        #     "lex": lex_code,
        #     "order": order_code
        # }
        # dd["w_wm_output_attacked"] = paraphrase_outputs
        dd[f"dipper_inputs_Lex{lex}_Order{order}"] = (final_input_text)
        dd['w_wm_output_attacked']=output_text.strip()

        return dd
        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    # data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)
    # data = data.add_column(f"dipper_inputs_Lex{lex}_Order{order}", dipper_inputs)


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
    lex=args.l #20
    order=args.o #20
    input_path = args.data_path
    output_path = os.path.splitext(input_path)[0] + f'_dipper_O{order}_L{lex}.jsonl'  # 修改输出路径

    # 获取上次处理到的行数
    processed_count = get_processed_count(output_path)
    prompt = "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
    # 读取输入文件

    time1 = time.time()
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    model = T5ForConditionalGeneration.from_pretrained("kalpeshk2011/dipper-paraphraser-xxl",device_map="auto")
    
    #model = T5ForConditionalGeneration.from_pretrained("~/.cache/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2",device_map="auto")
    print("Model loaded in ", time.time() - time1)
    # model.half()
    # model.cuda()
    model.eval()

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'a', encoding='utf-8') as outfile:  # 'a' 模式追加数据
        # 逐行处理，跳过已经处理的行
        for i, line in enumerate(tqdm(infile, desc="Processing samples", initial=processed_count)):
            if i < processed_count:  # 如果当前行已经处理过，则跳过
                continue

            data_item = json.loads(line.strip())  # 读取并解析每一行数据
            if data_item["w_wm_output_length"] < 50:
                print(data_item["w_wm_output_length"],"is too short, pass")
                continue
            updated_item = dipper_attacker(data_item,model=model,tokenizer=tokenizer,lex=100-lex,order=100-order,)
            outfile.write(json.dumps(updated_item, ensure_ascii=False) + "\n")
            
            # 每处理一个样本，更新已处理行数
            processed_count += 1
            update_processed_count(output_path, processed_count)

    print(f"Updated JSONL file saved to: {output_path}")

if __name__ == '__main__':
    main()
