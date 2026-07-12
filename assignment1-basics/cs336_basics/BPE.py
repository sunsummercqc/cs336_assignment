# vovabulary initialization

#Pre-tokenization: 
#Compute BPE merges
# Special tokens
from collections import Counter, defaultdict
import regex as re

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

##pre-tokenization 普通文本 → word_counts
def split_token_on_special_tokens(text, special_tokens):
    if not special_tokens:
        return pretokenize(text)
    pattern = "|".join(re.escape(token) for token in special_tokens)
    parts = re.split(pattern, text)
    total_word_counts = defaultdict(int)
    for part in parts:
        word_counts = pretokenize(part)
        for word, count in word_counts.items():
            total_word_counts[word] += count
    return total_word_counts

def pretokenize(text):
    
    word_counts = defaultdict(int)
    for match in re.finditer(GPT2_PATTERN, text):
       token = match.group(0)  #"the the cat" -> ["the", " the", " cat"]
       token_bytes = token.encode('utf-8') #"the" -> b'the' -> [b't', b'h', b'e']
       byte_word = tuple(bytes([b]) for b in token_bytes)  # "the" -> (b't', b'h', b'e')
    #    byte_word = tuple(bytes([b]) for b in word.encode('utf-8'))
       word_counts[byte_word] += 1
    return word_counts
        

    
def merge_word (word,pair):
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            result.append(pair[0] + pair[1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def build_pair_to_words(word_counts):
    pair_to_words = defaultdict(set)
    for word, count in word_counts.items():
        word_pair = get_word_pairs(word)
        #字典加某个元素的做法
        for pair in word_pair:
            pair_to_words[pair].add(word)
    return pair_to_words

def get_word_pairs(word):
    word_pairs = []
    i = 0
    while i < len(word) -1:
        word_pairs.append((word[i], word[i+1]))
        i+=1
    return word_pairs

def build_pair_counts(word_counts):
    pair_counts = Counter()
    for word, count in word_counts.items():
        for pair in get_word_pairs(word):
            pair_counts[pair] += count
    return pair_counts

def get_best_pair(pair_counts): #如果有多个best_pair呢
    best_pair = None
    max_count = 0
    for pair, count in pair_counts.items():
        if best_pair is None:
            best_pair = pair
            max_count = count
        elif count > max_count:
            max_count = count
            best_pair = pair
        elif count == max_count and pair > best_pair:
            best_pair = pair
    return best_pair

def word_has_pair(word, pair):
    i = 0
    while i < len(word) - 1 :
        if word[i] == pair[0] and word[i+1] == pair[1]:
            return True
        i+=1
    return False
# 把所有word里的best_pair都合并成一个token
def apply_merge(word_counts, pair):
    new_word_counts = defaultdict(int)
    for word, count in word_counts.items():
        ##optimize speed
        if word_has_pair(word,pair):
            new_word = merge_word(word,pair)
        else:
            new_word = word
        new_word_counts[new_word] += count
            
    return new_word_counts

def train_bpe_from_text(text, vocab_size, special_tokens):
    vocab = {}
    #print(len(vocab))
    for special_token in special_tokens:
        special_token = special_token.encode('utf-8')
        vocab[len(vocab)] = special_token
        #print(len(vocab))
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    merges = []
    word_counts = split_token_on_special_tokens(text, special_tokens)
    pair_counts = build_pair_counts(word_counts)
    pair_to_words = build_pair_to_words(word_counts)
    while len(vocab) < vocab_size:
        if len(pair_counts) == 0:
            break
        best_pair = get_best_pair(pair_counts)
        if best_pair is None or pair_counts[best_pair] <= 0:
            break
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)]= new_token

        affected_words = list(pair_to_words.get(best_pair, set()))
        updates = []
        for word in affected_words:
            count = word_counts.pop(word, None)
            if count is None or not word_has_pair(word, best_pair):
                continue

            for pair in get_word_pairs(word):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word)

            updates.append((merge_word(word, best_pair), count))

        for new_word, count in updates:
            word_counts[new_word] += count
            for pair in get_word_pairs(new_word):
                pair_counts[pair] += count
                pair_to_words[pair].add(new_word)
    return vocab, merges

def train_bpe(input_path, vocab_size, special_tokens):
    return train_bpe_from_text(open(input_path,'r', encoding='utf-8').read(),vocab_size,special_tokens)


# print(merge_word(['l', 'o', 'w'], ['l', 'o']))
# print(merge_word(['h', 'e', 'l', 'l', 'o'], ['l', 'l']))
# print(count_pairs({('l', 'o', 'w'): 1, ('l', 'o', 'w', 'e', 'r'): 1, ('n', 'e', 'w', 'e', 's', 't'): 1, ('w', 'i', 'd', 'e'): 1}))
#print(get_best_pair({('l', 'o'): 2, ('o', 'w'): 2, ('w', 'e'): 2, ('e', 'r')
# word_counts = {(b'l', b'o', b'w'): 1, (b'l', b'o', b'w', b'e', b'r'): 1, (b'n', b'e', b'w', b'e', b's', b't'): 1, (b'w', b'i', b'd', b'e'): 1}
# pair = (b'l', b'o')
# print(apply_merge(word_counts, pair))
if __name__ == "__main__":
    # word_counts = {
    #     (b"l", b"o", b"w"): 5,
    #     (b"l", b"o", b"w", b"e", b"r"): 2,
    #     (b"n", b"e", b"w", b"e", b"s", b"t"): 6,
    #     (b"w", b"i", b"d", b"e", b"s", b"t"): 3,
    # }
    # merges = []
    # for _ in range(10):
    #     pair_counts = count_pairs(word_counts)
    #     best_pair = get_best_pair(pair_counts)
    #     merges.append(best_pair)
    #     word_counts = apply_merge(word_counts,best_pair)

    # print(word_counts)
    # print(merges)
    # text = "hello hello world!"
    # word_counts = pretokenize(text)
    # print(word_counts)
    # vocab, merges = train_bpe_from_text("hello hello world", 260)
    # print(merges)
    # for i in range(256, len(vocab)):
    #     print(i, vocab[i])
    # print(pretokenize("the the cat"))
    # text = "the the cat."
    # for match in re.finditer(GPT2_PATTERN, text):
    #     print("match object:", match)
    #     print("group(0):", repr(match.group(0))) # repr可以显示出空格， match.group(0)是匹配到的字符串
    # special_token测试
    vocab, merges = train_bpe_from_text("the<|endoftext|> the cat.", 260, ["<|endoftext|>"])
    print(vocab[0])
    print(vocab[1])
    print(vocab[256])
    print(len(vocab))
    word_counts = split_token_on_special_tokens("the<|endoftext|> the cat.", ["<|endoftext|>"])
    print(word_counts)

