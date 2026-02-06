import unicode_data

class Tokenizer:
    def __init__(self):
        # 词表：id -> bytes
        self.vocab = {}
        # 合并规则：(bytes, bytes) -> id
        self.merges = {}
        # 特殊 token 映射
        self.special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError
