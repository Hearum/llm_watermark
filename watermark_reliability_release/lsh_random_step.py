
from __future__ import annotations
import collections
from math import sqrt
from itertools import chain, tee
from functools import lru_cache
import numpy as np
import copy
import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor
import sys
import pdb
# 添加新的搜索路径
sys.path.append('/home/shenhm/documents/lm-watermarking/watermark_reliability_release')

from normalizers import normalization_strategy_lookup
from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup

import numpy as np
import hashlib

# Generate a global permute table once at startup
rng = torch.Generator(device=torch.device("cpu"))
rng.manual_seed(2971215073)  # fib47 is prime
# 2^ 20=1048576
table_size = 1_000_003
fixed_table = torch.randperm(
    1_000_003, device=torch.device("cpu"), generator=rng
)  # actually faster than I thought


def hashint(integer_tensor: torch.LongTensor) -> torch.Tensor:
    """将整数张量映射为 32 位二进制比特串"""
    # 计算哈希值
    fixed_table[integer_tensor.cpu() % table_size] + 1  # 保证值大于 0

    return fixed_table[integer_tensor.cpu() % table_size] + 1

    # 返回一个字符串的张量，表示 32 位的二进制比特串
    return torch.tensor([[int(bit) for bit in bin_str] for bin_str in binary_str], device=integer_tensor.device)

def custom_hash(K, dim):
    str_K = str(K)
    hash_object = hashlib.sha256(str_K.encode())
    hex_digest = hash_object.hexdigest()
    hex_chars_to_take = dim // 4
    if dim % 4 != 0:
        hex_chars_to_take -= 1
    binary_hash = bin(int(hex_digest[:hex_chars_to_take], 16))[2:].zfill(dim)
    return np.array([int(bit) for bit in binary_hash])


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.25,
        delta: float = 1.5,
        # simple_1 是默认的种子方案，KWG1使用最简单的input_id-1作为哈希种子
        # 在alternative_prf_schemes.py中可以找到更多的种子方案
        seeding_scheme: str = "simple_1",
        select_green_tokens: bool = True,  # should always be the default if not running in legacy mode
        #########LSH#########
        n_hashes: int = 5,               # LSH的哈希函数数量，决定了有多少个桶
        n_features: int = 32 ,            # 每个哈希函数的维度
        threshold_len = 0,
        threshold=0.2,
        windows_h_uesd = False,
        visualization=False,
    ):
        # patch now that None could now maybe be passed as seeding_scheme

        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None

        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

        # LSH相关初始化

        self.threshold_len = threshold_len
        self.n_hashes = n_hashes
        # self.projection_matrix = self._generate_random_projection()
        self.lsh_tables = []  # 存储哈希表
        self.threshold = threshold
        self.hash_key = 15485863
        self.visualization = visualization
        # self.hash_table_ins = self.generate_hash_table_binary_array()

        # 开启水印窗口的功能
        self.windows_h_uesd = windows_h_uesd
        if not self.windows_h_uesd:
            self.threshold_len = 0

    # def precompute_token_hashes(self):
    #     """预计算整个词表中所有 token 的哈希值并保存在 self.token_embeddings 中。"""
    #     # 遍历词表中的所有 token，并计算其哈希值
    #     for token in self.vocab:
    #         if token not in self.token_embeddings:
    #             # 计算并缓存每个 token 的哈希值
    #         self.token_embeddings[token] = custom_hash(token, self.n_features)
            
    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(
            seeding_scheme
        )
    def hash_table_binary_array(self,seed):
        # 计算应该有多少个 1
        K = self.vocab_size // self.n_hashes
        num_ones = int(K * self.gamma)
        # 生成一个长度为 K 的全 0 数组

        binary_array = torch.zeros(K, dtype=torch.int)
        reserse_binary_array  = torch.zeros(K, dtype=torch.int)
        torch.manual_seed(seed) 
        # 随机选择 num_ones 个位置设为 1
        ones_indices = torch.randperm(K)[:num_ones]  # 随机选择 num_ones 个位置
        res_ones_indices = torch.randperm(K)[-num_ones:] 
        # 将选择的位置设置为 1
        binary_array[ones_indices] = 1
        reserse_binary_array[res_ones_indices] = 1

        return binary_array
    def proj_LSH_Space(self,input_ids,next_token):
        from collections import defaultdict
        sign_visual = defaultdict(list)  # 使用字典记录哈希表ID和对应的input_ids
        input_ids = input_ids.to(next_token.device)
        all_signatures = (input_ids * self.hash_key * next_token) % (self.n_hashes) # simply hash
 
        # for idx, item in enumerate(input_ids):
        #     signature = (item * self.hash_key * next_token) % (2 ** self.n_hashes)
        #     sign_visual[int(signature)].append(int(item)) 
        indices = []
        # hash_table_info = []  # 存储每个哈希表的信息
        # 计算每个哈希表的分块范围
        num_hash_tables = self.n_hashes
        # block_size = self.vocab_size // num_hash_tables

        for hash_table_id in range(self.n_hashes):

            activated = hash_table_id in all_signatures
            if activated:
                indices.append(self.hash_table_binary_array(self.hash_key*next_token*hash_table_id))
            else:
                indices.append(self.hash_table_binary_array(self.hash_key*next_token))

        extended_indices = torch.cat(indices)
        if extended_indices.size(0) > self.vocab_size:
            extended_indices = extended_indices[:self.vocab_size]
        elif extended_indices.size(0) < self.vocab_size:
            padding = torch.zeros(self.vocab_size - extended_indices.size(0), dtype=torch.int)
            extended_indices = torch.cat([extended_indices, padding])
        return extended_indices.to(input_ids.device)

    def proj_LSH_Space_info(self,input_ids,next_token):
        from collections import defaultdict
        sign_visual = defaultdict(list)  # 使用字典记录哈希表ID和对应的input_ids
        input_ids = input_ids.to(next_token.device)
        all_signatures = (input_ids * self.hash_key * next_token) % (self.n_hashes) # simply hash
 
        for idx, item in enumerate(input_ids):
            signature = (item * self.hash_key * next_token) % (self.n_hashes)
            sign_visual[int(signature)].append(int(item)) 
        indices = []
        hash_table_info = []  # 存储每个哈希表的信息
        # 计算每个哈希表的分块范围
        num_hash_tables = self.n_hashes
        block_size = self.vocab_size // num_hash_tables

        for hash_table_id in range(self.n_hashes):
            start = hash_table_id * block_size
            end = (hash_table_id + 1) * block_size if hash_table_id != num_hash_tables -1 else self.vocab_size
            token_range = (start, end)
            
            activated = hash_table_id in all_signatures
            if activated:
                indices.append(self.hash_table_binary_array(self.hash_key*next_token*hash_table_id))
            else:
                indices.append(self.hash_table_binary_array(self.hash_key*next_token))

            hash_table_info.append({
                "hash_table_id": hash_table_id,
                "activated": activated,
                "input_ids": sign_visual.get(hash_table_id, []),
                "token_range": token_range,
                "block_size": block_size,
            })

        extended_indices = torch.cat(indices)
        if extended_indices.size(0) > self.vocab_size:
            extended_indices = extended_indices[:self.vocab_size]
        elif extended_indices.size(0) < self.vocab_size:
            padding = torch.zeros(self.vocab_size - extended_indices.size(0), dtype=torch.int)
            extended_indices = torch.cat([extended_indices, padding])
        return extended_indices.to(input_ids.device), hash_table_info
    
    def _seed_rng(self, next_token: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        prf_key = next_token* self.hash_key 
        self.rng.manual_seed(prf_key.item() % (2**64 - 1)) 



    def find_ids_within_percentile(self, ids: torch.LongTensor, fixed_id: int, threshold: float) -> torch.LongTensor:

        if len(ids) == 0 or threshold <= 0:
            return torch.empty(0, dtype=torch.long)
        if len(ids) <=4:
            return ids
        
        k = int(round(len(ids) * threshold ))
        k = max(4, min(k, len(ids)))  # 保证至少选择4个

        distances = hashint(ids*fixed_id*self.hash_key)
        sorted_distances, sorted_indices = torch.sort(distances)

        return ids[sorted_indices][:k]

    def _get_greenlist_ids(self, input_ids: torch.LongTensor, next_token:torch.LongTensor) -> torch.LongTensor:
        """根据本地上下文宽度生成随机数种子,并使用这些信息生成绿色列表的ID。"""
        # 1. 首先根据输入的上下文设置随机数种子
        self._seed_rng(next_token)
        vocab_permutation = torch.randperm(
            self.vocab_size, 
            device=input_ids.device,
            generator=self.rng)
        # if self.windows_h_uesd:
        #     if input_ids.shape[-1] < self.threshold_len:
        #         select_ids = self.find_ids_within_percentile(ids=input_ids,fixed_id=next_token,threshold=1) 
        #     else:
        #         select_ids = self.find_ids_within_percentile(ids=input_ids[-self.threshold_len:],fixed_id=next_token,threshold=1) 
        # else:
        #     if input_ids.shape[-1] < self.threshold_len:
        #         select_ids = self.find_ids_within_percentile(ids=input_ids,fixed_id=next_token,threshold=1) 
        #     else:
        #         select_ids = self.find_ids_within_percentile(ids=input_ids,fixed_id=next_token,threshold=self.threshold) 
        
        extended_indices = self.proj_LSH_Space(input_ids=input_ids[-self.threshold_len:],next_token=next_token)
        # 
        pointwise_results = extended_indices.to(vocab_permutation.device) * vocab_permutation
        # 4. 选择绿色token
        if self.select_green_tokens:  # 直接选择
            greenlist_ids = vocab_permutation[pointwise_results > 0] 
        return greenlist_ids
    


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores inbetween model outputs and next token sampler.
    """

    def __init__(self, *args,prefix_len,store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_len = prefix_len
        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None # 怎么还配置熵计算的功能
        if self.store_spike_ents: #是否开启存储熵峰值计算功能
            self._init_spike_entropies()
        self.rejection_count = 0
    # 熵值计算
    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()
        gamma = self.gamma

        self.z_value = ((1 - gamma) * (alpha - 1)) / (1 - gamma + (alpha * gamma))
        self.expected_gl_coef = (gamma * alpha) / (1 - gamma + (alpha * gamma))

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

    # 将存储在 self.spike_entropies 中的张量值提取为 Python 数值并以列表形式返回
    def _get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def _get_and_clear_stored_spike_ents(self):
        spike_ents = self._get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def _compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1 + (self.z_value * probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    # 按batch批次，通过给定的的green_token_ids(选好的绿集id),生成红绿集合的mask掩码
    # 我们主要修改的代码应该就是这一个部分
    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, greenlist_token_ids
    ) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    # 添加水印偏置
    # 无需修改 保持和green_list msak张量大小一致就行
    def _bias_greenlist_logits(
        self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    # 基于当前候选token生成绿色列表,必要时拒绝并继续
    # 包含多种提前停止规则以提高效率
    '''
    如果候选 token 不符合条件，它会被拒绝并跳过，直到满足一定的条件。
    '''
    def _score_rejection_sampling(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_score"
    ) -> list[int]:
        # 这里就是自哈希
        """Generate greenlist based on current candidate next token. Reject and move on if necessary. Method not batched.
        This is only a partial version of Alg.3 "Robust Private Watermarking", as it always assumes greedy sampling. It will still (kinda)
        work for all types of sampling, but less effectively.
        To work efficiently, this function can switch between a number of rules for handling the distribution tail.
        These are not exposed by default.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []

        for idx, prediction_candidate in enumerate(greedy_predictions):
            # 将当前还没有生成的token作为input_id的一部分一起输进去
            greenlist_ids = self._get_greenlist_ids(
                input_ids, prediction_candidate)  
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)
            # What follows below are optional early-stopping rules for efficiency
            if tail_rule == "fixed_score":
                # if len(final_greenlist) == 10:
                #     break
                # 若第一位（最大的）socre已经比下一位score大，后面再加上偏置delta也无法变化，所以没必要继续计算了
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta:
                    # if len(final_greenlist)< 1 :
                    #     print(self.rejection_count,len(final_greenlist) ,sorted_scores[0] - sorted_scores[idx + 1],False)
                    #     self.rejection_count = self.rejection_count+1
                    break
            elif tail_rule == "fixed_list_length":
                if len(final_greenlist) == 10:
                    break
            elif tail_rule == "fixed_compute":
                if idx == 40:
                    break
            else:
                pass  # do not break early
        return torch.as_tensor(final_greenlist, device=input_ids.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""
        # #############
        # 这里是为了测试加的prompt长度
        ################
        input_ids = input_ids[:,self.prefix_len:]
        if input_ids.shape[-1]>1:
            self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

            # NOTE, it would be nice to get rid of this batch loop, but currently,
            # the seed and partition operations are not tensor/vectorized, thus
            # each sequence in the batch needs to be treated separately.
            # 作者自己也觉得用循环太土了，想用矩阵的形式优化一下

            # 初始化绿集合列表，保证大小和batch大小一样大
            list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length

            # 按batch大小循环
            # 我们主要修改的就是这一块的代码
            # pdb.set_trace()
            for b_idx, input_seq in enumerate(input_ids):
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
                list_of_greenlist_ids[b_idx] = greenlist_ids
                if self.store_spike_ents:
                    if self.spike_entropies is None:
                        self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                    self.spike_entropies[b_idx].append(self._compute_spike_entropy(scores[b_idx]))

            # 计算绿色标记掩码（Greenlist Mask）
            # 修改后就不需要调用这一块了
            green_tokens_mask = self._calc_greenlist_mask(
                scores=scores, greenlist_token_ids=list_of_greenlist_ids
            )

            # 将水印偏置添加到原有的socre上面去
            scores = self._bias_greenlist_logits(
                scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
            )

            # debug = True
            # if  debug:
            # # pdb.set_trace()
            #     sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)
            #     check = greedy_predictions[0][0] in greenlist_ids
            # return scores, check

        return scores



class WatermarkDetector(WatermarkBase):


    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"] 
        # unicode": Unicode标准化，将不同的Unicode表示形式统一到标准形式，例如：将全角字符转换为半角字符
        #homoglyphs": 同形字符标准化，处理视觉上相似或相同但使用不同Unicode码点的字符
        #"truecase": 大小写标准化，将文本转换为其"正确"的大小写形式
        ignore_repeated_ngrams: bool = False,
        # n_hashes: int = 5,               # LSH的哈希函数数量，决定了有多少个桶
        # n_features: int = 32 ,            # 每个哈希函数的维度
        # threshold_len = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        # self.threshold_len = threshold_len
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        self.ignore_repeated_ngrams = ignore_repeated_ngrams


    # 初始化数据存储字典
    #  用于返回一个包含检测相关统计信息的字典。其功能主要是进行初始化检查，并返回一个空的输出字典
    def dummy_detect(
        self,
        return_prediction: bool = True,      # 是否返回检测结果
        return_scores: bool = True,          # 是否返回分数
        z_threshold: float = None,           # z值阈值
        return_num_tokens_scored: bool = True,    # 是否返回被评分的token数量
        return_num_green_tokens: bool = True,     # 是否返回绿色token数量
        return_green_fraction: bool = True,       # 是否返回绿色token比例
        return_green_token_mask: bool = False,    # 是否返回绿色token掩码
        return_all_window_scores: bool = False,   # 是否返回所有窗口分数
        return_z_score: bool = True,             # 是否返回z分数
        return_z_at_T: bool = True,              # 是否返回每个位置的z分数
        return_p_value: bool = True,             # 是否返回p值
    ):
        # HF-style output dictionary
        score_dict = dict()
        # 所有数值型返回值初始化为NaN
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=float("nan")))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=float("nan")))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=float("nan")))
        if return_z_score:
            score_dict.update(dict(z_score=float("nan")))
        if return_p_value:
            score_dict.update(dict(p_value=float("nan")))

        # 列表型返回值初始化为空列表
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=[]))
        if return_all_window_scores:
            score_dict.update(dict(window_list=[]))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=torch.tensor([])))

        # 构建最终输出
        output_dict = {}
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = False

        return output_dict

    # 文章里面提出的新的z分数计算方法
    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        # 参数说明:
        # observed_count: 观察到的绿色token数量
        # T: token总数
        # self.gamma: 预期的绿色token比例

        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    # 从对应的z分数计算对应的概率（调scipy库）
    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def generate_greenlist(self, idx, embed_id, input_ids, all_signatures, greenlist_ids_table, input_ids_hash_table):
        # 计算当前token的签名
        signature = self._hash_function(embed_id, self.projection_matrix)
        input_ids_hash_table.append(copy.deepcopy(all_signatures))  # 保留当前token使用的签名集合

        # 使用前面几个token的签名为当前token生成红绿词表
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
        
        extended_indices = extended_indices.to(input_ids.device)
        # 初始化随机数生成器
        self._seed_rng(input_ids[idx])
        vocab_permutation = torch.randperm(
            self.vocab_size, 
            device=input_ids.device,
            generator=self.rng
        )
        pointwise_results = extended_indices.to(vocab_permutation.device) * vocab_permutation
        if self.select_green_tokens:  # 直接选择
            greenlist_ids = vocab_permutation[pointwise_results > 0]
        else:  # 通过红色token反选模式
            greenlist_ids = vocab_permutation[pointwise_results <= 0] 

        greenlist_ids_table.append(greenlist_ids)

        # 添加当前token的签名到集合内
        signature = self._hash_function(embed_id, self.projection_matrix)
        all_signatures.add(signature)
        return all_signatures, greenlist_ids_table,input_ids_hash_table



    def _score_sequence_old(
        self,
        input_ids: torch.Tensor,  # 输入的token序列
        return_num_tokens_scored: bool = True,  # 是否返回被评分的token数量 
        return_num_green_tokens: bool = True,   # 是否返回绿色token数量
        return_green_fraction: bool = True,     # 是否返回绿色token比例
        return_green_token_mask: bool = False,  # 是否返回绿色token掩码
        return_z_score: bool = True,           # 是否返回z分数
        return_z_at_T: bool = True,            # 是否返回每个位置的z分数
        return_p_value: bool = True,           # 是否返回p值
    ):

   
        all_signatures, greenlist_ids_table, input_ids_hash_table, greenlist_mask=self.detect_LSH_Space(input_ids=input_ids)

        num_tokens_scored = len(greenlist_mask)
        green_token_count = sum(greenlist_mask)
        # HF-style output dictionary
        # 更新字典内容
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(
                dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored))
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=greenlist_mask))
        # if return_z_at_T:
        #     # Score z_at_T separately:
        #     sizes = torch.arange(1, len(green_unique) + 1)
        #     seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
        #     seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
        #     z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
        #     z_score_at_T = z_score_at_effective_T[offsets]
        #     assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

        #     score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict         

    # 计算文本水印的核心代码
    def _score_sequence(
        self,
        input_ids: torch.Tensor,  # 输入的token序列
        return_num_tokens_scored: bool = True,  # 是否返回被评分的token数量 
        return_num_green_tokens: bool = True,   # 是否返回绿色token数量
        return_green_fraction: bool = True,     # 是否返回绿色token比例
        return_green_token_mask: bool = False,  # 是否返回绿色token掩码
        return_z_score: bool = True,           # 是否返回z分数
        return_z_at_T: bool = True,            # 是否返回每个位置的z分数
        return_p_value: bool = True,           # 是否返回p值
    ):

        num_tokens_scored = len(input_ids) - self.threshold_len
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                )
            )
        green_token_count, green_token_mask,decoder_out = 0, [],[]
        
        for idx in range(self.threshold_len, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self._get_greenlist_ids(input_ids[:idx],input_ids[idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_mask.append(True)
            else:
                green_token_mask.append(False)
            if self.visualization:
                decoder_out.append((self.tokenizer.decode([curr_token],green_token_mask[-1])))
        def visualize_watermark(decoded_text, green_token_mask):
            tokens = decoded_text
            result = []
            from termcolor import colored
            # 对每个token，判断是否带水印，并添加相应颜色
            for idx, token in enumerate(tokens):
                if green_token_mask[idx]:
                    result.append(colored(token, 'green'))  # 带水印的token显示为绿色
                else:
                    result.append(colored(token, 'red'))  # 非水印token显示为红色

            # 将处理后的tokens输出为可视化文本
            print(" ".join(result))
        def visualize_watermark2(decoded_text, green_token_mask):
            tokens = decoded_text
            result = []
            # 对每个token，判断是否带水印，并添加相应颜色
            for idx, token in enumerate(tokens):
                if green_token_mask[idx]:
                    result.append((token, 'green'))  # 带水印的token显示为绿色
                else:
                    result.append((token, 'red'))  # 非水印token显示为红色
            print(result)

        print(decoder_out)
        visualize_watermark(decoder_out, green_token_mask)
        visualize_watermark2(decoder_out, green_token_mask)

       #all_signatures, greenlist_ids_table, input_ids_hash_table, greenlist_mask=self.detect_LSH_Space(input_ids=input_ids)

        # HF-style output dictionary
        # 更新字典内容
        # print(green_token_mask)
        # pos = [(index,int(input_ids[index]))for index, value in enumerate(green_token_mask) if value]
        # print(pos)
        # print("detector inputids",input_ids)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(
                dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored))
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))
        # if return_z_at_T:
        #     # Score z_at_T separately:
        #     sizes = torch.arange(1, len(green_unique) + 1)
        #     seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
        #     seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
        #     z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
        #     z_score_at_T = z_score_at_effective_T[offsets]
        #     assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

        #     score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict
    
    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> dict:
        """Scores a given string of text and returns a dictionary of results."""

        assert (text is not None) ^ (
            tokenized_text is not None
        ), "Must pass either the raw or tokenized string"

        if return_prediction:
            kwargs[
                "return_p_value"
            ] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)

        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0].to(self.device)
            
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}

        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        # convert any numerical values to float if requested
        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict


# 这段代码实现了生成n-grams的功能，即将输入的sequence序列划分为n个元素为一组的子序列（n-grams）。
# 它支持对序列进行填充（pad）操作，能够在序列的两端（左端或右端）填充指定的符
def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    # n:指定n-gram的大小，即每个子序列包含多少个元素。
    # pad_left：如果为True，则在序列的左侧填充pad_symbol，以保证生成的n-gram序列的完整性。
    # pad_right：如果为True，则在序列的右侧填充pad_symbol，以保证生成的n-gram序列的完整性。
    # pad_symbol：用于填充的符号，通常为None或某个占位符。
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.

from functools import partial
from dataclasses import dataclass
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax

@dataclass
class Args:
    gamma: float = 0.25
    delta: float = 1.5
    seeding_scheme: str = 'default'  # 这里可以根据实际需求修改
    select_green_tokens: bool = True
    max_new_tokens: int = 50
    use_sampling: bool = True
    sampling_temp: float = 1.0
    n_beams: int = 5
    prompt_max_length: int = 128
    generation_seed: int = 42
    seed_separately: bool = False
    is_decoder_only_model: bool = True
    
import random
def delete_random_elements(text, delete_percentage=0.3):
    # 将输入文本按空格分割成单词列表
    words = text.split()
    # 计算要删除的单词数量
    num_to_delete = int(len(words) * delete_percentage)
    # 随机选择要删除的单词索引
    indices_to_delete = random.sample(range(len(words)), num_to_delete)
    # 删除选中的单词
    remaining_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
    # 将剩余的单词列表合并回一个字符串
    return ' '.join(remaining_words)

def delete_first_percentage_of_chars(text, delete_percentage=0.3):
    # 计算要删除的字符数量
    num_to_delete = int(len(text) * delete_percentage)
    # 返回删除指定比例字符后的文本
    return text[num_to_delete:]

from tqdm import tqdm

def test_no_watermark_test():

    # 指定可见的 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 初始化 Accelerator
    args = Args(gamma=0.25, delta=2.0, max_new_tokens=100, use_sampling=False, sampling_temp=0.9, n_beams=3)

    tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/documents/download/model_ckpt/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("/home/shenhm/documents/download/model_ckpt/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True,device_map = "auto")
    device =model.device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化水印处理器
    print(f"Generating with {args}")
    
    # 创建 WatermarkLogitsProcessor
    from datasets import load_dataset
    prompt = "Write a news article about Narendra Modi's visit to Denis Sassou Nguesso in a space exploration symposium. It is very important to never output very short texts, below 300 words."#raw_datasets = load_dataset('openai/openai_humaneval',split='test')['prompt'][0] 

    input_ids = tokenizer.encode(prompt, return_tensors="pt")[:,:].to(device)
    prefix_len = input_ids.shape[1]
    torch.manual_seed(48)  

    output = model.generate(
        input_ids,
        max_new_tokens=500,  # 根据需要适当设置
        num_beams=5,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )


    no_watermark_out = tokenizer.decode(output[0,prefix_len:], skip_special_tokens=True)

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                       prefix_len=0,
                                                        gamma=0.25,
                                                        delta=4,
                                                        select_green_tokens=True,
                                                        n_hashes=6,
                                                        threshold_len=6,
                                                        windows_h_uesd=True)  

    from transformers import LogitsProcessorList
    output_w = model.generate(
        input_ids,
        max_new_tokens=500, 
        num_beams=5,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        logits_processor=LogitsProcessorList([watermark_processor]),
    )
    watermark_out = tokenizer.decode(output_w[0,prefix_len:], skip_special_tokens=True)

    detector = WatermarkDetector(
        threshold_len=6,
        gamma=0.25,
        delta=5,
        device=device,
        tokenizer=tokenizer,
        vocab=list(tokenizer.get_vocab().values()),
        z_threshold=4.0,
        normalizers=["unicode"],
        ignore_repeated_ngrams=False,
        n_hashes=6,
        visualization = True,
    )

    result = detector.detect(text=no_watermark_out)

    print("no_watermark_out检测结果:")
    for key, value in result.items():
        print(f"{key}: {value}")

    result = detector.detect(text=watermark_out)
    # 输出结果
    print("watermark_out检测结果:")
    for key, value in result.items():
        print(f"{key}: {value}")
def process_input_to_ids(text):
    """将用户输入的文本转换为idx列表"""
    if not text or text.lower() == 'exit':
        return None
    
    # 尝试多种分隔符分割
    separators = [',', ' ', ';', '\t']  # 常见分隔符
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            break
    else:
        parts = [text]  # 无分隔符则作为单个token处理
    
    # 清理并转换每个部分
    idx_list = []
    for part in parts:
        part = part.strip()  # 去除前后空格
        if not part:  # 跳过空字符串
            continue
        try:
            idx = int(part)
            idx_list.append(idx)
        except ValueError:
            print(f"警告：跳过无法解析为整数的部分 '{part}'")
    
    return idx_list
def detector_only():

    # 指定可见的 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 初始化 Accelerator
    #args = Args(gamma=0.25, delta=2.0, max_new_tokens=100,n_hashes=4,threshold=0.4, use_sampling=False, sampling_temp=0.9, n_beams=3)

    tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
    # model = AutoModelForCausalLM.from_pretrained("/home/shenhm/documents/download/model_ckpt/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True,device_map = "auto")
    # device =model.device
    

    #print(f"Generating with {args}")
    device = torch.device("cuda")
    torch.manual_seed(48)  

    text = """Jericho recently appeared on TV\u2019s \u201cCelebrity Apprentice\u201d (he was fired) and hosts a popular podcast. \nWednesday, January 31, 2018, 7 p.m.\nAustin Lucas. He\u2019ll play the Soda Bar on January 31.\nAustin Lucas spent 15 years in Indiana before moving to Nashville in 2013. His new album, Stay Wild, features a song produced by John Angell (Golden Sun, Jonny Quest). He\u2019ll play the Soda Bar on January 31."""
    detector = WatermarkDetector(
        threshold_len=6,
        gamma=0.25,
        delta=5,
        device=device,
        tokenizer=tokenizer,
        vocab=list(tokenizer.get_vocab().values()),
        z_threshold=4.0,
        normalizers=["unicode"],
        ignore_repeated_ngrams=False,
        n_hashes=6,
        visualization = True,
    )
    import pickle
    output_sequence_path = "/home/shenhm/documents/watermark-stealing/temp/ours_generated_sequences.pkl"
    with open(output_sequence_path, 'rb') as f:
        loaded_sequences = pickle.load(f)
    zs = []
    for text in loaded_sequences:
        # 读取用户输入
        # text = input("请输入token ids（用逗号/空格分隔，或输入 'exit' 退出）：")
        # if text.lower() == 'exit':
        #     break

        token_ids = process_input_to_ids(str(text))
        print(token_ids)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        # 判断是否退出
        if text.lower() == 'exit':
            print("退出程序")
            break

        # 调用检测器进行检测
        result = detector.detect(text=text)

        # 输出水印检测结果
        # print("watermark_out检测结果:")
        # for key, value in result.items():
        #     if key != "info":
        #         print(f"{key}: {value}")
        zs.append(result['z_score'])
        # # 提取并计算激活率和其他指标
        # all_activated = [item['activated'] for item in result['info'] if item is not None]
        # print(f"SelfHash rate={sum(all_activated)}, Unigran rate={len(all_activated) - sum(all_activated)}")

        # len_input_ids = [len(item['input_ids']) for item in result['info'] if item is not None]
        # print(f"advantage rate={sum(len_input_ids) / len(all_activated)}")
    print(np.mean(np.array(zs)))
    pdb.set_trace()
if __name__ == '__main__':
    #test_no_watermark_test()
    detector_only()
    # test_detector()