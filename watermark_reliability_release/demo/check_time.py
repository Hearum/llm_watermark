import pstats

# 加载 .prof 文件
p = pstats.Stats('/home/shenhm/documents/output.prof')

# 打印简单的统计信息
p.strip_dirs()  # 去掉路径中的多余部分，便于查看
p.sort_stats('cumtime').print_stats(100)


def proj_LSH_Space(self,input_ids: torch.LongTensor,):
    all_signatures = set()
    embed_ids = hashint_to_bin(input_ids) 
    for embed_id in embed_ids:
        signature = self._hash_function(embed_id, self.projection_matrix)
        all_signatures.add(signature)
    indices = []
    for hash_table_id in range(2**self.n_hashes):
        if hash_table_id in all_signatures:
            indices.append(self.hash_table_ins[hash_table_id]["binary_array"])
        else:
            indices.append(self.hash_table_ins[hash_table_id]["reserse_binary_array"])

    extended_indices = torch.cat(indices)
    if extended_indices.size(0) > self.vocab_size:
        extended_indices = extended_indices[:self.vocab_size]
    elif extended_indices.size(0) < self.vocab_size:
        padding = torch.zeros(self.vocab_size - extended_indices.size(0), dtype=torch.int)
        extended_indices = torch.cat([extended_indices, padding])
    return extended_indices.to(input_ids.device)


def _seed_rng(self, next_token: torch.LongTensor) -> None:
    """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
    prf_key = next_token* self.hash_key 
    self.rng.manual_seed(prf_key.item() % (2**64 - 1)) 
    

def _get_greenlist_ids(self, input_ids: torch.LongTensor, next_token:torch.LongTensor) -> torch.LongTensor:
    """根据本地上下文宽度生成随机数种子,并使用这些信息生成绿色列表的ID。"""
    
    # 1. 首先根据输入的上下文设置随机数种子
    self._seed_rng(next_token)

    vocab_permutation = torch.randperm(
        self.vocab_size, 
        device=input_ids.device,
        generator=self.rng)
    
    extended_indices = self.proj_LSH_Space(input_ids=input_ids)
    pointwise_results = extended_indices.to(vocab_permutation.device) * vocab_permutation
    
    # 4. 选择绿色token
    if self.select_green_tokens:  # 直接选择
        greenlist_ids = vocab_permutation[pointwise_results > 0] 
    else:  # 通过红色token反选模式
        greenlist_ids = vocab_permutation[pointwise_results <= 0] 
    return greenlist_ids