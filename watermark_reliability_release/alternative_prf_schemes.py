"""Implement other PRF functions, so, hashing schemes.

Can be hooked into existing WatermarkLogitsProcessor as modified base class WatermarkBase
"""

import torch
from itertools import combinations
from functools import cache

# Key properties of a hashing scheme
props = {
    "prf_type": str,  # string name of the underlying PRF mapping multiple token ids to a random seed
    "context_width": int,  # this is h in the paper, how many previous tokens should be considered for each PRF
    "self_salt": bool,  # Use the rules laid in robust-watermarking to use the token itself to seed and possibly reject its own list
    "hash_key": int,  # integer, large prime, used to move seed away from low-entrop bit sequences in PRF chosen above
}


def seeding_scheme_lookup(seeding_scheme: str):
    if not isinstance(seeding_scheme, str):
        raise ValueError("Seeding scheme should be a string summarizing the procedure.")
    if seeding_scheme == "simple_1" or seeding_scheme == "lefthash":
        # Default, simple bigram hash  # alias for ff-additive_prf-1-False-15485863
        prf_type = "additive_prf" # KWG1 Lefthash
        context_width = 1
        self_salt = False
        hash_key = 15485863
    elif seeding_scheme == "algorithm-3" or seeding_scheme == "selfhash":
        prf_type = "anchored_minhash_prf"
        context_width = 4
        self_salt = True
        hash_key = 15485863
    elif seeding_scheme == "skipgram":
        prf_type = "skipgram_prf"
        context_width = 5
        self_salt = False
        hash_key = 15485863
    elif seeding_scheme.startswith(
        "ff"
    ):  # freeform seeding scheme API - only use for experimenting
        # expects strings of the form ff-additive_prf-4-True-hash or ff-additive_prf-5-True (hash key is optional)
        # 自由形式种子方案API，只用于实验
        # KWG2.0里面使用的最好的4位selfhash方案，seeding_scheme="ff-anchored_minhash_prf-4-True-15485863
        split_scheme = seeding_scheme.split("-")
        prf_type = str(split_scheme[1]) # anchored_minhash_prf
        context_width = int(split_scheme[2]) # 4
        self_salt = split_scheme[3] == "True" #True
        if len(split_scheme) == 5:
            hash_key = int(split_scheme[4])
        else: # ICML这篇文章使用的密钥为15485863
            hash_key = 15485863
    else:
        raise ValueError(f"Invalid seeding scheme name {seeding_scheme} given. Try  'simple_1'?")

    assert prf_type in prf_lookup.keys()
    return prf_type, context_width, self_salt, hash_key

# salt_key是PRF的种子密钥（虽然这里写作加盐值）
# 计算输入token ID的乘积，并将其与salt_key相乘，生成一个整数种子。
def multiplicative_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.prod().item()

# 计算输入token ID的总和，并将其与salt_key相乘，生成一个整数种子。
def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()

# 找到输入token ID中的最小值，并将其与salt_key相乘，生成一个整数种子。
def minfunc_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    # not a great idea for non-random input ids as in text
    return salt_key * input_ids.min().item()

# 计算输入token ID的k-skipgram的乘积，并将其与salt_key相乘，生成一个整数种子。
# k是跳跃距离，默认值为2
# 按步长k选择输入token ID中的部分元素，乘以salt_key，然后对这些结果应用hashint函数，最后计算这些哈希值的乘积，生成一个整数种子
def simple_skip_prf(input_ids: torch.LongTensor, salt_key: int, k=2) -> int:
    # k is the skip distance
    return hashint(salt_key * input_ids[::k]).prod().item()

# 原文中的leftHash
# 仅使用输入序列中的第一个token ID，乘以salt_key，然后应用hashint函数，生成一个整数种子。
def skipgram_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    # maximum distance skipgram within context
    return hashint(salt_key * input_ids[0]).item()

# 使用第一个和最后一个id作为hash key
# 结合输入序列中的第一个token ID和指定位置（默认是最后一个token ID）的token ID，乘以salt_key，分别应用hashint函数，然后将两个哈希值相乘，生成一个整数种子。
def anchored_skipgram_prf(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # maximum distance skipgram within context
    return (hashint(salt_key * input_ids[0]) * hashint(salt_key * input_ids[anchor])).item()

# 原文中的minHash
# 使用最小值的哈希
def minhash_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return hashint(salt_key * input_ids).min().item()

# 本质上也是一种minHash,和上面那个不同的点在于，这里是先做hash再找最小值，在steal里面想的方法是一样的(实际上用的好像也是这个)
def anchored_minhash_prf(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * hashint(input_ids) * hashint(input_ids[anchor])).min().item()

def anchored_minhash_prf4(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * myhashint_4(input_ids) * hashint(input_ids[anchor])).min().item()

def anchored_minhash_prf8(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * myhashint_8(input_ids) * hashint(input_ids[anchor])).min().item()

def anchored_minhash_prf6(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * myhashint_6(input_ids) * hashint(input_ids[anchor])).min().item()

def anchored_minhash_prf16(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * myhashint_16(input_ids) * hashint(input_ids[anchor])).min().item()

def anchored_minhash_prf32(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * myhashint_32(input_ids) * hashint(input_ids[anchor])).min().item()

def anchored_minhash_prf64(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * myhashint_64(input_ids) * hashint(input_ids[anchor])).min().item()

# 生成所有可能的跳跃gram（skipgrams），即从token ID集合中选择两个token ID的组合，应用hashint函数后计算每对的乘积，然后找到这些乘积中的最小值，生成一个整数种子。
def minskipgram_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    # min over all skipgrams in context, k=2 is all pairs
    skipgrams = torch.as_tensor(list(combinations(hashint(salt_key * input_ids), 2)))
    return skipgrams.prod(dim=1).min().item()


def noncomm_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    key = torch.as_tensor(salt_key, dtype=torch.long)
    for entry in input_ids:
        key *= hashint(key * entry)
        key %= 2**32
    return key.item()


def position_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    return (
        (salt_key * input_ids * torch.arange(1, len(input_ids) + 1, device=input_ids.device))
        .sum()
        .item()
    )

# 定义PRF函数中使用的种子策略
prf_lookup = {
    "multiplicative_prf": multiplicative_prf,
    "additive_prf": additive_prf,
    "minfunc_prf": minfunc_prf,
    "simple_skip_prf": simple_skip_prf,
    "skipgram_prf": skipgram_prf,
    "anchored_skipgram_prf": anchored_skipgram_prf,
    "minhash_prf": minhash_prf,
    "anchored_minhash_prf": anchored_minhash_prf,
    "anchored_minhash_prf4": anchored_minhash_prf4,
    "anchored_minhash_prf6": anchored_minhash_prf6,
    "anchored_minhash_prf8": anchored_minhash_prf8,
    "anchored_minhash_prf16": anchored_minhash_prf16,
    "anchored_minhash_prf32": anchored_minhash_prf32,
    "anchored_minhash_prf64": anchored_minhash_prf64,
    "minskipgram_prf": minskipgram_prf,
    "noncomm_prf": noncomm_prf,
    "position_prf": position_prf,
}

# Generate a global permute table once at startup
rng = torch.Generator(device=torch.device("cpu"))
rng.manual_seed(2971215073)  # fib47 is prime
table_size = 1_000_003
fixed_table = torch.randperm(
    1_000_003, device=torch.device("cpu"), generator=rng
)  # actually faster than I thought

# 用于将输入的整数张量转换为另一个整数张量，类似于哈希操作（本质上就是最简单的一种哈希，设置一个足够大的table_size然后取模)
# 利用一个预定义的固定查找表（fixed_table）对输入进行映射，从而实现一种简单且高效的哈希功能。
def hashint(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % table_size] + 1
    )  # minor cheat here, this function always return CPU values
def myhashint_6(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % 6] + 1
    )  # minor cheat here, this function always return CPU values


def myhashint_4(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % 4] + 1
    )  # minor cheat here, this function always return CPU values
    
def myhashint_8(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % 8] + 1
    )  # minor cheat here, this function always return CPU values

def myhashint_16(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % 16] + 1
    )  # minor cheat here, this function always return CPU values
def myhashint_32(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % 32] + 1
    )  # minor cheat here, this function always return CPU values

def myhashint_64(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % 64] + 1
    )  # minor cheat here, this function always return CPU values
def _hashint_avalanche_tensor(integer_tensor: torch.LongTensor):
    """http://burtleburtle.net/bob/hash/integer.html, ported into pytorch, runs on tensors. Apparently a decent avalanche."""
    i = integer_tensor.to(torch.int32).clone()  # or torch.int16?
    i -= i << 6
    i ^= i >> 17
    i -= i << 9
    i ^= i << 4
    i -= i << 3
    i ^= i << 10
    i ^= i >> 15
    return i.to(torch.long)


@cache
def _hashint_avalanche_int(integer: int):
    """http://burtleburtle.net/bob/hash/integer.html, runs in base python, caches based on access.
    Does this make sense for signed 64bit ints?"""
    i = integer % (2**32)
    i -= i << 6
    i ^= i >> 17
    i -= i << 9
    i ^= i << 4
    i -= i << 3
    i ^= i << 10
    i ^= i >> 15
    return i
