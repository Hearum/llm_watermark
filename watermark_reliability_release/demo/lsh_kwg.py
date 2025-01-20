# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt
from itertools import chain, tee
from functools import lru_cache

import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from normalizers import normalization_strategy_lookup
from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        # simple_1 是默认的种子方案，KWG1使用最简单的input_id-1作为哈希种子
        # 在alternative_prf_schemes.py中可以找到更多的种子方案
        seeding_scheme: str = "simple_1",  # simple default, find more schemes in alternative_prf_schemes.py

        select_green_tokens: bool = True,  # should always be the default if not running in legacy mode
    ):
        # patch now that None could now maybe be passed as seeding_scheme
        if seeding_scheme is None:
            seeding_scheme = "simple_1"

        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self._initialize_seeding_scheme(seeding_scheme)
        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(
            seeding_scheme
        )

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # 不支持批处理，因为我们使用的生成器(如cuda.random)不支持批处理。
        # 如果没有指定seeding_scheme，使用实例的默认值

        # 1. 检查上下文长度
        # 需要有足够的上下文用于种子生成
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG."
            )
        
        # 2. 生成PRF密钥
        # 这里的prf_lookup是一个字典，存储了各种的方法，从这里将密钥hash_key和对应要保护的上下文长度hash_key输入即可
        prf_key = prf_lookup[self.prf_type](
            input_ids[-self.context_width :], salt_key=self.hash_key   # 使用最后context_width个token
        )

        # 3. 设置随机种子
        # 对prf_key取模以防止溢出(最大值为2^64-1)，随机数种子生成器有上限值
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """根据本地上下文宽度生成随机数种子,并使用这些信息生成绿色列表的ID。"""
        
        # 1. 首先根据输入的上下文设置随机数种子
        self._seed_rng(input_ids)

        # 2. 计算绿色列表的大小
        # gamma是一个比例系数(0-1之间),vocab_size是词表大小
        # greenlist_size表示要选择的绿色token数量
        greenlist_size = int(self.vocab_size * self.gamma)
        
        # 3. 生成词表大小的随机排列
        # randperm会生成一个0到vocab_size-1的随机排列
        vocab_permutation = torch.randperm(
            self.vocab_size, 
            device=input_ids.device,
            generator=self.rng
        )
        
        # 4. 选择绿色token
        if self.select_green_tokens:  # 直接选择模式
            # 从随机排列的开头选择greenlist_size个token作为绿色token
            greenlist_ids = vocab_permutation[:greenlist_size]
        else:  # 通过红色token反选模式(旧的行为)
            # 从随机排列的末尾选择greenlist_size个token作为绿色token 
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size):]
        
        return greenlist_ids

# 封装在Hunggingface logitsprocessor的水印添加类实习爱你
class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores inbetween model outputs and next token sampler.
    """

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None # 怎么还配置熵计算的功能
        if self.store_spike_ents: #是否开启存储熵峰值计算功能
            self._init_spike_entropies()

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
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute"
    ) -> list[int]:
        # 这里就是自哈希的描述吧
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
                torch.cat([input_ids, prediction_candidate[None]], dim=0)
            )  # add candidate to prefix
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)

            # What follows below are optional early-stopping rules for efficiency
            if tail_rule == "fixed_score":
                # 若第一位（最大的）socre已经比下一位score大，后面再加上偏置delta也无法变化，所以没必要继续计算了
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta:
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

        # this is lazy to allow us to co-locate on the watermarked model's device
        # 初始化随机数生成器
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        # 作者自己也觉得用循环太土了，想用矩阵的形式优化一下

        # 初始化绿集合列表，保证大小和batch大小一样大
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length

        # 按batch大小循环
        # 我们主要修改的就是这一块的代码
        for b_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                # 两种都是词表划分方法，上面这种好像是用了自哈希的方案+提前终止采样
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            # 获取当前batch_id下的绿集合词表
            list_of_greenlist_ids[b_idx] = greenlist_ids

            # 计算和存储尖峰熵
            # 用于可视化和后续讨论用的
            # logic for computing and storing spike entropies for analysis
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

        return scores


class WatermarkDetector(WatermarkBase):
    """This is the detector for all watermarks imprinted with WatermarkLogitsProcessor.

    The detector needs to be given the exact same settings that were given during text generation  to replicate the watermark
    greenlist generation and so detect the watermark.
    This includes the correct device that was used during text generation, the correct tokenizer, the correct
    seeding_scheme name, and parameters (delta, gamma).

    Optional arguments are
    * normalizers ["unicode", "homoglyphs", "truecase"] -> These can mitigate modifications to generated text that could trip the watermark
    * ignore_repeated_ngrams -> This option changes the detection rules to count every unique ngram only once.
    * z_threshold -> Changing this threshold will change the sensitivity of the detector.

    该检测器需要提供与文本生成时完全相同的设置，以便复制水印绿色列表的生成，从而检测水印。这些设置包括生成时使用的正确设备、
    正确的分词器、正确的 seeding_scheme 名称以及相关参数 delta、gamma。
    normalizers ["unicode", "homoglyphs", "truecase"] -> 这些可以缓解生成文本中的修改，避免干扰水印检测。
    ignore_repeated_ngrams -> 该选项修改检测规则,只统计每个唯一的n-gram一次。
    z_threshold -> 改变该阈值将调整检测器的敏感度。
    """

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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

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

    # 缓存N-gram分数，避免了多次重复计算
    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int):
        """Expensive re-seeding and sampling is cached."""
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        return True if target in greenlist_ids else False
    
    # 该方法从输入的文本中提取所有n-gram，并计算它们的水印得分。它使用滑动窗口的方式来提取n-gram，并计算每个n-gram的得分，判断最后一个token是否属于“绿色列表”。
    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )

        # Compute scores for all ngrams contexts in the passage:
        token_ngram_generator = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt
        )
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(prefix, target)

        return ngram_to_watermark_lookup, frequencies_table
    # 生成绿色/红色Token掩码（_get_green_at_T_booleans）
    # 该方法生成一个二进制掩码，标记文本中的每个token是否为绿色（包含水印）或红色（不包含水印）。如果启用了ignore_repeated_ngrams，则重复的n-gram不会重复计数。
    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup) -> tuple[torch.Tensor]:
        """Generate binary list of green vs. red per token, a separate list that ignores repeated ngrams, and a list of offsets to
        convert between both representations:
        green_token_mask = green_token_mask_unique[offsets] except for all locations where otherwise a repeat would be counted
        """
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
        )

    # 计算文本水印的核心代码
    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )

        # Count up scores over all ngrams
        if self.ignore_repeated_ngrams:
            # Method that only counts a green/red hit once per unique ngram.
            # New num total tokens scored (T) becomes the number unique ngrams.
            # We iterate over all unqiue token ngrams in the input, computing the greenlist
            # induced by the context in each, and then checking whether the last
            # token falls in that greenlist.
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )
        assert green_token_count == green_unique.sum()

        # HF-style output dictionary
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
            score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        if return_z_at_T:
            # Score z_at_T separately:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

            score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict

    def _score_windows_impl_batched(
        self,
        input_ids: torch.Tensor,
        window_size: str,
        window_stride: int = 1,
    ):
        # Implementation details:
        # 1) --ignore_repeated_ngrams is applied globally, and windowing is then applied over the reduced binary vector
        #      this is only one way of doing it, another would be to ignore bigrams within each window (maybe harder to parallelize that)
        # 2) These windows on the binary vector of green/red hits, independent of context_width, in contrast to Kezhi's first implementation
        # 3) z-scores from this implementation cannot be directly converted to p-values, and should only be used as labels for a
        #    ROC chart that calibrates to a chosen FPR. Due, to windowing, the multiple hypotheses will increase scores across the board#
        #    naive_count_correction=True is a partial remedy to this

        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_mask, green_ids, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )
        len_full_context = len(green_ids)

        partial_sum_id_table = torch.cumsum(green_ids, dim=0)

        if window_size == "max":
            # could start later, small window sizes cannot generate enough power
            # more principled: solve (T * Spike_Entropy - g * T) / sqrt(T * g * (1 - g)) = z_thresh for T
            sizes = range(1, len_full_context)
        else:
            sizes = [int(x) for x in window_size.split(",") if len(x) > 0]

        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        s = window_stride

        window_fits = False
        for idx, size in enumerate(sizes):
            if size <= len_full_context:
                # Compute hits within window for all positions in parallel:
                window_score = torch.zeros(len_full_context - size + 1, dtype=torch.long)
                # Include 0-th window
                window_score[0] = partial_sum_id_table[size - 1]
                # All other windows from the 1st:
                window_score[1:] = partial_sum_id_table[size::s] - partial_sum_id_table[:-size:s]

                # Now compute batched z_scores
                batched_z_score_enum = window_score - self.gamma * size
                z_score_denom = sqrt(size * self.gamma * (1 - self.gamma))
                batched_z_score = batched_z_score_enum / z_score_denom

                # And find the maximal hit
                maximal_z_score = batched_z_score.max()
                z_score_max_per_window[idx] = maximal_z_score

                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::s] = torch.maximum(
                    cumulative_eff_z_score[size::s], z_score_at_effective_T[:-1]
                )
                window_fits = True  # successful computation for any window in sizes

        if not window_fits:
            raise ValueError(
                f"Could not find a fitting window with window sizes {window_size} for (effective) context length {len_full_context}."
            )

        # Compute optimal window size and z-score
        cumulative_z_score = cumulative_eff_z_score[offsets]
        optimal_z, optimal_window_size_idx = z_score_max_per_window.max(dim=0)
        optimal_window_size = sizes[optimal_window_size_idx]
        return (
            optimal_z,
            optimal_window_size,
            z_score_max_per_window,
            cumulative_z_score,
            green_mask,
        )

    def _score_sequence_window(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        window_size: str = None,
        window_stride: int = 1,
    ):
        (
            optimal_z,
            optimal_window_size,
            _,
            z_score_at_T,
            green_mask,
        ) = self._score_windows_impl_batched(input_ids, window_size, window_stride)

        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=optimal_window_size))

        denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma))
        green_token_count = int(optimal_z * denom + self.gamma * optimal_window_size)
        green_fraction = green_token_count / optimal_window_size
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=green_fraction))
        if return_z_score:
            score_dict.update(dict(z_score=optimal_z))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=z_score_at_T))
        if return_p_value:
            z_score = score_dict.get("z_score", optimal_z)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))

        # Return per-token results for mask. This is still the same, just scored by windows
        # todo would be to mark the actually counted tokens differently
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_mask.tolist()))

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

        if window_size is not None:
            # assert window_size <= len(tokenized_text) cannot assert for all new types
            score_dict = self._score_sequence_window(
                tokenized_text,
                window_size=window_size,
                window_stride=window_stride,
                **kwargs,
            )
            output_dict.update(score_dict)
        else:
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


##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
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
