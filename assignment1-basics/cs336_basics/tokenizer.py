# BPE training :input corpus -> vocab, merges
#Tokenizer : text -> token ids
import regex as re 
from collections import defaultdict
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from typing import Iterator

from collections import defaultdict
class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        #对每个token做re.escape
        self.special_pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
        self.byte_to_id = {v: k for k,v in vocab.items()}
        self.merge_ranks = {pair : i for i, pair in enumerate(merges)}
    def encode(self, text) -> list[int]:
        token_id = []
        if len(self.special_tokens) == 0:
            for chunk in self.pretokenize(text):
                pieces = self.apply_merges(chunk)
                token_id.extend([self.byte_to_id[p] for p in pieces])
        else:
            #pattern = "|".join(re.escape(token) for token in self.special_tokens)
            parts = re.split(self.special_pattern, text)
            for part in parts:
                if part is None:
                    continue
                if part in self.special_tokens:
                    token_id.append(self.byte_to_id[part.encode('utf-8')])
                else:
                    for chunk in self.pretokenize(part):
                        pieces = self.apply_merges(chunk)
                        token_id.extend([self.byte_to_id[p] for p in pieces])
        return token_id

    # 按照merges的顺序合并
    def pretokenize(self, text):
        all_trunks = []
        for match in re.finditer(GPT2_PATTERN, text):
            token = match.group(0)
            token_bytes = token.encode('utf-8')
            chunk = []
            # byte_word = tuple(bytes([b]) for b in token_bytes)
            for b in token_bytes:
                chunk.append(bytes([b]))
            all_trunks.append(chunk)
        #print(f"DEBUG: pretokenize 返回的是: {all_trunks}")
        return all_trunks

    def apply_merges(self, pieces):
        #使用merge ranks快速查找优先级
        #遍历pieces中的pair, 获得其在merge_ranks里面的rank值，有则返回最小的rank，没有则返回inf
        # 得到排序后的pair，进行合并，并更新pair
        #print(f"Deubug: 当前pieces是：{pieces}")
        while len(pieces) > 1:
            #print(f"Deubug: 当前pieces是：{pieces}")
            pairs = [(pieces[i], pieces[i+1]) for i in range(len(pieces) - 1)]
            best_pair = min(pairs, key = lambda p : self.merge_ranks.get(p, float('inf'))) 
            # 若返回的pair的rank值为inf，要么该pair不合法要么已经合并到尽头
            if self.merge_ranks.get(best_pair, float('inf')) == float('inf'):
                break
            i = 0
            new_pieces = []
            while i < len(pieces):
                if i < len(pieces) - 1 and (pieces[i], pieces[i+1]) == best_pair:
                    new_pieces.append(pieces[i] + pieces[i+1])
                    i = i+2
                # 合并未发生，加入未合并的词
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = new_pieces

        return pieces


    def decode(self, token_ids) -> str:
        pieces =  [self.vocab [i] for i in token_ids]
        return b"".join(pieces).decode("utf-8", errors="replace")
    def encode_iterable(self, iterator) -> Iterator[int]:
        for chunk in iterator:
            yield from self.encode(chunk)
 
if __name__ == "__main__":
    # 1. 构造一个最简单的 BPE 模型 (仅用于测试 encode)
    # 假设我们只训练了把 'h'+'e' 合并为 'he'
    vocab = {0: b'h', 1: b'e', 2: b'l', 3: b'o', 4: b'he'}
    merges = [(b'h', b'e')] # 训练规则：先合并 h 和 e
    # 2. 实例化你的 Tokenizer
    tokenizer = BPETokenizer(vocab, merges)
    # 3. 执行测试
    print(tokenizer.encode("hello"))
