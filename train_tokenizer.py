from src.tokenizer import MiniBPETokenizer

# 1. 准备一小段文本
example_text = """
Hello world! This is a nano-llama project. 
The BPE tokenizer will merge frequent byte pairs.
你好，世界！这是一个微型大模型项目。
"""

# 2. 初始化并训练
tokenizer = MiniBPETokenizer()
# 256(基础字节) + 20(合并次数) = 276 词表大小
tokenizer.train(example_text, vocab_size=276, verbose=True)

# 3. 测试编码和解码
test_str = "Hello project! 你好世界"
ids = tokenizer.encode(test_str)
decoded_str = tokenizer.decode(ids)

print(f"\nOriginal: {test_str}")
print(f"IDs: {ids}")
print(f"Decoded: {decoded_str}")
assert test_str == decoded_str, "Decode error!"
