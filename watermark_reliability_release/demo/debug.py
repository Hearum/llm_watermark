import torch
import time
import torch

# 假设 vocab_indices 和 pointwise_results 都是 torch 张量
vocab_indices = torch.tensor([1, 2, 3, 4, 5, 6])
pointwise_results = torch.tensor([0, 1, -1, 3, 0, 2])

# 第一步：筛选出 pointwise_results > 0 的元素
green_candidates = vocab_indices[pointwise_results > 0]

# 第二步：获取 green_candidates 中大于 0 的位置
greater_than_zero_positions = (green_candidates > 0).nonzero(as_tuple=True)[0]

# 打印结果
print("green_candidates:", green_candidates)
print("Positions in green_candidates greater than 0:", greater_than_zero_positions)

# def generate_binary_array(token, K, gamma):
#     # 计算应该有多少个 1
#     num_ones = int(K * gamma)
    
#     # 生成一个长度为 K 的全 0 数组
#     binary_array = torch.zeros(K, dtype=torch.int)
    
#     # 用 token 来决定哪些位置是 1
#     # 使用 token 的哈希值作为种子，生成一个随机序列
#     torch.manual_seed(abs(hash(token)) % (2**32))  # 使用 token 的哈希值作为种子
    
#     # 随机选择 num_ones 个位置设为 1
#     ones_indices = torch.randperm(K)[:num_ones]  # 随机选择 num_ones 个位置
    
#     # 将选择的位置设置为 1
#     binary_array[ones_indices] = 1
    
#     return binary_array

# # 测试函数的执行时间
# def test_speed():
#     token = "example_token"
#     K = 1000  # 数组长度
#     gamma = 0.3  # 1 的比例
    
#     # 测量时间
#     start_time = time.time()
    
#     # 执行多次生成二进制数组的操作
#     for _ in range(1000):  # 运行 1000 次来测试性能
#         generate_binary_array(token, K, gamma)
    
#     end_time = time.time()
    
#     elapsed_time = end_time - start_time
#     print(f"Elapsed time for 1000 iterations: {elapsed_time:.4f} seconds")
    
# # 调用测试函数
# test_speed()



# import numpy as np
# import hashlib
# def custom_hash(K, dim):
#     str_K = str(K)
#     hash_object = hashlib.sha256(str_K.encode())
#     hex_digest = hash_object.hexdigest()
#     hex_chars_to_take = dim // 4
#     if dim % 4 != 0:
#         hex_chars_to_take -= 1
#     binary_hash = bin(int(hex_digest[:hex_chars_to_take], 16))[2:].zfill(dim)
#     return np.array([int(bit) for bit in binary_hash])

# print(2**32)
# # 测试
# for n in range(10):
#     bitstring =  custom_hash(n,32)
#     print(n,bitstring)




# import nltk
# nltk.download('punkt_tab')

 # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)