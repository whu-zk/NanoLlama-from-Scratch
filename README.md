# NanoLlama
## Tokneizer

### 为什么使用字节级（Byte-level）BPE？
*   *回答：* 避免了 OOV（Out-of-vocabulary）问题，任何字符串都能通过 UTF-8 编码成字节流。

**为什么需要正则表达式预切割？**
    *   *回答：* 参考了 GPT-4/Llama 的做法。如果不加正则，Tokenizer 可能会把“word!”中的“d!”合并在一起，或者把跨越空格的字符合并。正则保证了语义单元（如单词、数字、缩写）的初步隔离，提高了编码效
