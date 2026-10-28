### Problem
1. OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。 Error loading "D:\桌面\self-learning\cs336\assignment1-basics\.venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
2. 激活环境：.venv\Scripts\activateuv sync
- 解决方法：
venv的环境又和conda环境混合在一起了,退出了base环境
配置cpu版本pytorch
- uv pip install --reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
查看版本
- python -c "import importlib.metadata as m; print(m.version('torch'))"
- python -c "import torch; print(torch.__version__)"
3. 数据文件无法正常下载
- 

  
### 5.7 
1. 了解作业配环境中，2h未完成，同时了解了一下要做什么项目，目前的计划是做cs336的作业1和5同步做一个特定领域的微调
### 5.8
开始执行测试文件
- uv run --no-sync pytest tests/test_nn_utils.py -q
### 5.9
1. 正式启动晚了BPE的5个函数，依靠ai的提示磕磕巴巴写完了，但是我还是很享受在自己思维的流淌下去写代码的感觉到的，而且没写对也能立即得到纠正，慢即是快，继续坚持啊啊！
2. 数据文件下载到50%就自动停了，好像只有没开代理的时候才能在镜像网站上继续下载
### 5.12
1. 执行第一个测试：uv run --no-sync pytest tests/test_train_bpe.py::test_train_bpe -q
- 在修修补补中获得了一个passed！！！
2.  uv run --no-sync pytest tests/test_train_bpe.py -q
- 速度未通过
- special token通过
- 速度未通过
### 5.13
1. 优化速度、完成tokenizer

### 5.18
1. 感觉自己好像傻子，根本学不懂，学不会，学了就忘，那些人怎么做到一学就会的

### 6.2 每日深度思考2h
1. 任务：train_bpe_tinystories
  - python -m cs336_basics.bpe_tinystories :启动训练
### 6.3
1. 重新理解了BPE
## BPE Tokenizaer training
1. utf-16的编码方式对应的字符串转义的长度更短
 - Encode:
    str（Unicode Code Points） → bytes（计算机可理解的语言、8位整数、每个元素是0-255之间的整数）
 - Decode
    bytes → str 
 - 基础知识
  1 Byte (字节) = 8 bits (位)，计算机存储一个英文字符通常占用 1 Byte。
  1 KB (Kilobyte) = 1024 Bytes。
  4 KB = $4 \times 1024 = 4096$ Bytes。
2. unicode encoding → word tokenization → subword tokenization
## bytes
| 类型      | 存储内容            | 示例                | 用途           |
| ------- | --------------- | ----------------- | ------------ |
| `str`   | 文本字符（Unicode）   | `"你好"`            | 给人看的文字       |
| `bytes` | 原始字节（0-255 的整数） | `b'\xe4\xbd\xa0'` | 给计算机处理的二进制数据 |
* 理解bytes
  
  bytes([228]) # b'\xe4' 表示一个二进制字符
  'byte_word = tuple(bytes([b]) for b in token_bytes) '
  token_bytes是 b'low',
  python遍历bytes 时，拿到的是每个 byte 的整数值,即b'l':108,b'0':111,b'w':119，因此此处的b是32，116等整数值, bytes[b]将整数转回包含一个字节的bytes对象，e.g. b'l'，这里的字符对应的数值是固定的，即固定的编码标准
  - 又不理解了

### 6.15
1. 完善tokenizer
   1. encode
   2. decode
   3. encode_iterator
### 6.16
1. 完善encode
2. min的另一种用法:筛选模式
   - min(iterable, key= ...)
     - iterable: pairs
     - key: 评价标准
    - e.g. 
      - # 这堆人 (pairs)
          people = [{"name": "小明", "age": 20}, {"name": "小红", "age": 18}, {"name": "小刚", "age": 25}]
          选出年龄最小的人
          这里的 p 就代表每一个人：{"name": "小明", "age": 20} 等等
          youngest = min(people, key=lambda p: p["age"])
          print(type(youngest))  # 输出: <class 'dict'>
          print(youngest["name"]) # 输出: 小红
3. extend 和 append的区别：
      1. list_a = [1, 2, 3]
          list_a.append([4, 5])
          print(list_a)
          # 输出: [1, 2, 3, [4, 5]]
          # 注意：[4, 5] 变成了一个嵌套的子列表
      2. list_b = [1, 2, 3]
          list_b.extend([4, 5])
          print(list_b)
          # 输出: [1, 2, 3, 4, 5]
          # 注意：4 和 5 被分别添加到列表中了
        ！注意：这两个方法只返回None
4. 字典获取索引的两个方法，dict[p] vs dict.get(p,default)
5. iterator: 
   - 什么是 Iterator (迭代器)？
    简单来说，Iterator 是一个可以记住“我走到哪儿了”的工具。
    当你有一个非常大的数据集（比如几个 GB 的文本），你不能一次性把它全部读进内存。这时候就需要Iterator：

  

6. GPT-2 regex 保留空格的分词
7. 加上special token的操作
special_token： list[str]
dict类型的数据不可以使用append
1. BPE整个算法流程
- BPE的作用是什么？
  - 若只是用词表记录所有的输入文本，造成极大内存占用，只用单个字符，序列会太长
- 流程：
  - text → GPT-2风格的小块 → UTF-8 bytes → bytes拆成单byte token序列 → 统计counts，得到best_pair  

- merge : 
- 重新统计word_count: {
    (b"t", b"h", b"e"): 3,
    (b" ", b"t", b"h", b"e"): 10,
    (b" ", b"c", b"a", b"t"): 2,
}
- 不熟悉的概念：
  - 区别vocab_size 和 token计数 ，vocab_size的数量即为一个llm可以处理的词表的长度
1. 若pair为tuple，pair ('b','h'), （pair[0]+pair[1]）和（pair[0],pair[1]）的区别是什么
- 不熟悉的字典用法 count = word_counts.pop(word, None)
- 不熟悉的set用法
- 
print(word_counts)
defaultdict(<class 'int'>, {(b't', b'h', b'e'): 1, (b' ', b't', b'h', b'e'): 1, (b' ', b'c', b'a', b't'): 1, (b'.',): 1})
print(pair_counts)
Counter({(b't', b'h'): 2, (b'h', b'e'): 2, (b' ', b't'): 1, (b' ', b'c'): 1, (b'c', b'a'): 1, (b'a', b't'): 1})
print(pair_to_words)
defaultdict(<class 'set'>, {(b't', b'h'): {(b't', b'h', b'e'), (b' ', b't', b'h', b'e')}, (b'h', b'e'): {(b't', b'h', b'e'), (b' ', b't', b'h', b'e')}, (b' ', b't'): {(b' ', b't', b'h', b'e')}, (b' ', b'c'): {(b' ', b'c', b'a', b't')}, (b'c', b'a'): {(b' ', b'c', b'a', b't')}, (b'a', b't'): {(b' ', b'c', b'a', b't')}})
print(new_token)
b'th'
print(vocab)
{0: b'<|endoftext|>', 1: b'\x00', 2: b'\x01', 3: b'\x02', 4: b'\x03', 5: b'\x04', 6: b'\x05', 7: b'\x06', 8: b'\x07', 9: b'\x08', 10: b'\t', 11: b'\n', 12: b'\x0b', 13: b'\x0c', 14: b'\r', 15: b'\x0e', 16: b'\x0f', 17: b'\x10', 18: b'\x11', 19: b'\x12', 20: b'\x13', 21: b'\x14', 22: b'\x15', 23: b'\x16', 24: b'\x17', 25: b'\x18', 26: b'\x19', 27: b'\x1a', 28: b'\x1b', 29: b'\x1c', 30: b'\x1d', 31: b'\x1e', 32: b'\x1f', 33: b' ', 34: b'!', 35: b'"', 36: b'#', 37: b'$', 38: b'%', 39: b'&', 40: b"'", 41: b'(', 42: b')', 43: b'*', 44: b'+', 45: b',', 46: b'-', 47: b'.', 48: b'/', 49: b'0', 50: b'1', 51: b'2', 52: b'3', 53: b'4', 54: b'5', 55: b'6', 56: b'7', 57: b'8', 58: b'9', 59: b':', 60: b';', 61: b'<', 62: b'=', 63: b'>', 64: b'?', 65: b'@', 66: b'A', 67: b'B', 68: b'C', 69: b'D', 70: b'E', 71: b'F', 72: b'G', 73: b'H', 74: b'I', 75: b'J', 76: b'K', 77: b'L', 78: b'M', 79: b'N', 80: b'O', 81: b'P', 82: b'Q', 83: b'R', 84: b'S', 85: b'T', 86: b'U', 87: b'V', 88: b'W', 89: b'X', 90: b'Y', 91: b'Z', 92: b'[', 93: b'\\', 94: b']', 95: b'^', 96: b'_', 97: b'`', 98: b'a', 99: b'b', 100: b'c', 101: b'd', 102: b'e', 103: b'f', 104: b'g', 105: b'h', 106: b'i', 107: b'j', 108: b'k', 109: b'l', 110: b'm', 111: b'n', 112: b'o', 113: b'p', 114: b'q', 115: b'r', 116: b's', 117: b't', 118: b'u', 119: b'v', 120: b'w', 121: b'x', 122: b'y', 123: b'z', 124: b'{', 125: b'|', 126: b'}', 127: b'~', 128: b'\x7f', 129: b'\x80', 130: b'\x81', 131: b'\x82', 132: b'\x83', 133: b'\x84', 134: b'\x85'}
print(bytes([134]))
b'\x86'
print(merges)
[(b't', b'h')]
print(affected_words)
[(b't', b'h', b'e'), (b' ', b't', b'h', b'e')]
1. 调试技巧
若代码陷入某个断点的函数循环出不来，可以让鼠标到达下一个断点，然后右键选择jump to cursor
    
### 10.9

## 注意力机制
注意力机制就是
Q ： 
K
V 
attention = Q * K
## 多头注意力机制

