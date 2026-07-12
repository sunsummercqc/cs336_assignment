import time
import tracemalloc
import pickle
from cs336_basics.BPE import train_bpe 

# 1. 开启内存监控
tracemalloc.start()


start_time = time.time()

# 2. 设置参数
input_path = "./tests/fixtures/tinystories_sample.txt" 
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

# 3. 开始训练
vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
print(vocab)
print(merges)

# 4. 记录结束状态
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# 5. 保存结果
with open("vocab.pkl", "wb") as f: pickle.dump(vocab, f)
with open("merges.pkl", "wb") as f: pickle.dump(merges, f)

print(f"训练耗时: {end_time - start_time:.2f} 秒")
print(f"内存占用峰值: {peak / 10**6:.2f} MB")

# 假设 vocab 是你训练后得到的字典，结构为 {token_id: b'token_bytes'}
def find_longest_token(vocab):
    # 找到 vocab 中 bytes 长度最长的那个值
    # key=len 会告诉 max 函数去比较每个 bytes 对象的长度
    longest_token_bytes = max(vocab.values(), key=len)
    
    # 同时找到对应的 token_id (如果需要的话)
    longest_token_id = -1
    for token_id, token_bytes in vocab.items():
        if token_bytes == longest_token_bytes:
            longest_token_id = token_id
            break
            
    return longest_token_id, longest_token_bytes

# 使用方法
longest_id, longest_bytes = find_longest_token(vocab)
print(f"最长 Token 的 ID 是: {longest_id}")
print(f"最长 Token 的内容 (bytes): {longest_bytes}")
print(f"最长 Token 的长度: {len(longest_bytes)} 字节")
# 尝试解码看看它是什么字符
print(f"解码后的字符串: {longest_bytes.decode('utf-8', errors='replace')}")