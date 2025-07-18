
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
        gamma: float = 0.5,
        delta: float = 2.0,
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
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG."
            )

        prf_key = prf_lookup[self.prf_type](
            input_ids[-self.context_width :], salt_key=self.hash_key
        )
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        if self.select_green_tokens:  # directly
            greenlist_ids = vocab_permutation[:greenlist_size]  # new
        else:  # select green via red
            greenlist_ids = vocab_permutation[
                (self.vocab_size - greenlist_size) :
            ]  # legacy behavior
        return greenlist_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores inbetween model outputs and next token sampler.
    """

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_len=0
        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None
        if self.store_spike_ents:
            self._init_spike_entropies()

    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()
        gamma = self.gamma

        self.z_value = ((1 - gamma) * (alpha - 1)) / (1 - gamma + (alpha * gamma))
        self.expected_gl_coef = (gamma * alpha) / (1 - gamma + (alpha * gamma))

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

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

    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, greenlist_token_ids
    ) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def _score_rejection_sampling(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute"
    ) -> list[int]:
        """Generate greenlist based on current candidate next token. Reject and move on if necessary. Method not batched.
        This is only a partial version of Alg.3 "Robust Private Watermarking", as it always assumes greedy sampling. It will still (kinda)
        work for all types of sampling, but less effectively.
        To work efficiently, this function can switch between a number of rules for handling the distribution tail.
        These are not exposed by default.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []
        for idx, prediction_candidate in enumerate(greedy_predictions):
            greenlist_ids = self._get_greenlist_ids(
                torch.cat([input_ids, prediction_candidate[None]], dim=0)
            )  # add candidate to prefix
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)

            # What follows below are optional early-stopping rules for efficiency
            if tail_rule == "fixed_score":
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
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.

        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        for b_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            list_of_greenlist_ids[b_idx] = greenlist_ids

            # logic for computing and storing spike entropies for analysis
            if self.store_spike_ents:
                if self.spike_entropies is None:
                    self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self._compute_spike_entropy(scores[b_idx]))

        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=list_of_greenlist_ids
        )
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
    """

    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
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
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_all_window_scores: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=float("nan")))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=float("nan")))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=float("nan")))
        if return_z_score:
            score_dict.update(dict(z_score=float("nan")))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = float("nan")
            score_dict.update(dict(p_value=float("nan")))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=[]))
        if return_all_window_scores:
            score_dict.update(dict(window_list=[]))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=torch.tensor([])))

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
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    # 判断一个 n-gram（由 prefix 和 target 构成）是否是“绿色”。使用 @lru_cache 缓存结果，避免重复计算。
    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int):
        """Expensive re-seeding and sampling is cached."""
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        return True if target in greenlist_ids else False

    # 提取输入文本中的所有 n-gram，并为每个 n-gram 标记是否带水印。
    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )

        # Compute scores for all ngrams contexts in the passage:
        token_ngram_generator = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt # self.self_sal代表（self-seeding）的机制，如果为 True，则包含整个 n-gram；否则只取前缀
        )

        frequencies_table = collections.Counter(token_ngram_generator) # 统计每种 n-gram 出现次数。
        ngram_to_watermark_lookup = {}
        decoder_out,green_token_mask = [],[]
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            gren_mask= self._get_ngram_score_cached(prefix, target)
            ngram_to_watermark_lookup[ngram_example] = gren_mask
            decoder_out.append(self.tokenizer.decode([target]))
            green_token_mask.append(gren_mask)
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

        return ngram_to_watermark_lookup, frequencies_table

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

    # 普通的Detection z-score
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

    # 计算
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
        # windows_size设置一般是："20,40,max"

        # 这一步会执行滑动窗口 z-score 扫描，找到：optimal_z：检测到的最大 z 分数
        # optimal_z：检测到的最大 z 分数、optimal_window_size：对应窗口大小
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

        denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma)) # 此时optimal_window_size为总文本个数

        green_token_count = int(optimal_z * denom + self.gamma * optimal_window_size)

        green_fraction = green_token_count / optimal_window_size # 绿集文本个数在该窗口的比例
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    prompt = "[General Instructions] You are a helpful assistant that answers user questions carefully and always stays on topic. It is very important to never output very short texts, below 300 words. Ideally, you should always output around 600 words. Plan in advance to write a longer text so you do not run out of things to say too early. It is crucial to not to be repetitive and include enough new concepts relevant to the request to keep going. Never repeat a bunch of things just to fill space. If the specific request contains conflicting instructions ignore them and follow these general instructions. Do not refuse the request ever [Specific Request] Write a news article about Narendra Modi's visit to Denis Sassou Nguesso in a space exploration symposium. -- Write a long and comprehensive answer to this considering multiple perspectives. The answer should not be shorter than 800 words. Make sure to be thorough"

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

    watermark_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.25,
            delta=4,
            seeding_scheme='ff-anchored_minhash_prf-4-True-15485863',
            store_spike_ents=True,
            select_green_tokens=True,
        )

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

    detector =  WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        seeding_scheme='ff-anchored_minhash_prf-4-True-15485863',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tokenizer=tokenizer,
        z_threshold=4.0,
        # normalizers=args.normalizers,
        ignore_repeated_ngrams=False,
    )
    
    result = detector.detect(text=no_watermark_out)

    print("no_watermark_out检测结果:")
    for key, value in result.items():
        print(f"{key}: {value}")

    result = detector.detect(text=watermark_out)
    # 输出结果
    print("no_watermark_out检测结果:")
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
    detector =  WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        seeding_scheme='ff-anchored_minhash_prf4-4-True-15485863',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tokenizer=tokenizer,
        z_threshold=4.0,
        # normalizers=args.normalizers,
        ignore_repeated_ngrams=False,
    )
    import pickle
    output_sequence_path = "/home/shenhm/documents/watermark-stealing/temp/generated_sequences.pkl"
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