
import os
import json
import torch
import pdb

data_path = "/home/shenhm/documents/watermark-stealing/out_llama/1/ours/c4-kgw-ff-anchored_minhash_prf-16-True-15485863/0.jsonl"

with open(data_path,'r') as file:
    data = [json.loads(line.strip()) for line in file]
# dict_keys(['idx', 'prompt', 'textwm'])

pdb.set_trace()