import torch
import numpy as np
import hashlib

import numpy as np
import hashlib

# Generate a global permute table once at startup
rng = torch.Generator(device=torch.device("cpu"))
rng.manual_seed(2971215073)  # fib47 is prime
# 2^ 20=1048576
table_size = 4294967295
fixed_table = torch.randperm(
    4294967295, device=torch.device("cpu"), generator=rng
)  # actually faster than I thought

# 用于将输入的整数张量转换为另一个整数张量，类似于哈希操作（本质上就是最简单的一种哈希，设置一个足够大的table_size然后取模)
# 利用一个预定义的固定查找表（fixed_table）对输入进行映射，从而实现一种简单且高效的哈希功能。
def hashint_to_bin(integer_tensor: torch.LongTensor) -> torch.Tensor:
    """将整数张量映射为 32 位二进制比特串"""
    # 计算哈希值
    hash_value = fixed_table[integer_tensor.cpu() % table_size] + 1  # 保证值大于 0
    
    # 将哈希值转换为 32 位的二进制字符串
    # 使用 format 将整数转换为 32 位二进制格式，确保长度为 32 位
    binary_str = [format(val.item(), '032b') for val in hash_value]

    # 返回一个字符串的张量，表示 32 位的二进制比特串
    return torch.tensor(binary_str,device=integer_tensor.device)
# 模拟哈希投影的函数
class WatermarkBase:
    def __init__(self, projection_matrix=None, n_hashes=4, n_features=32):
        # 这里我们简单地使用随机生成的投影矩阵
        self.n_hashes = n_hashes
        self.n_features = n_features
        self.projection_matrix = torch.randn(n_features, n_hashes)

    def _hash_function(self, point, projection_matrix):
        """基于随机投影生成哈希值"""
        projected = torch.matmul(point, projection_matrix)  # 投影计算
        # 大于等于 0 的部分为 1，否则为 0
        return torch.ge(projected, 0).int()  # 返回 0 或 1

    def project_next_token(self, input_ids: torch.LongTensor, next_token: torch.LongTensor):
        """通过LSH生成签名"""
        all_signatures = set()
        
        # 通过 custom_hash 获取输入的哈希签名
        embed_ids = hashint_to_bin(input_ids.numpy())  # custom_hash 需要的是 numpy 数组
        
        # 对每个哈希值进行映射
        for embed_id in embed_ids:
            # 将 embed_id 变成 tensor 来适配 _hash_function
            signature = self._hash_function(torch.tensor(embed_id).float(), self.projection_matrix)
            
            # 将二进制签名转为十进制
            binary_signature = ''.join(map(str, signature.tolist()))
            decimal_signature = int(binary_signature, 2)
            
            # 将十进制签名加入集合
            all_signatures.add(decimal_signature)

        return all_signatures

# 测试用例
if __name__ == "__main__":
    # 示例 input_ids 和 next_token
    input_ids = torch.LongTensor([1234, 5678, 91011])  # 假设输入的 LongTensor 类型
    next_token = torch.LongTensor([12])  # 假设下一个 token 是 12

    # 初始化 WatermarkBase 类，投影矩阵维度为 32，哈希空间数为 4
    watermark = WatermarkBase(n_hashes=4, n_features=32)

    # 调用 project_next_token
    all_signatures = watermark.project_next_token(input_ids, next_token)

    # 输出结果
    print("生成的所有签名（十进制）：")
    for sig in all_signatures:
        print(sig)
