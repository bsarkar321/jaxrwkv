from tokenizers import Tokenizer
from pyrwkv_tokenizer import RWKVTokenizer

import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class BaseTokenizer:
    def encode(self, src):
        raise NotImplementedError("Implement encode in the subclass")

    def decode(self, tokens):
        raise NotImplementedError("Implement decode in the subclass")

    def pad_encode(self, src, pad_alignment=16):
        encoded = self.encode(src)
        return_length = len(encoded)
        if return_length % pad_alignment != 0:
            encoded = encoded + [0] * (pad_alignment - return_length % pad_alignment)
        return encoded, return_length


class GptTokenizer(BaseTokenizer):

    def __init__(self):
        self.tok = Tokenizer.from_file(os.path.join(dir_path, "tok_files/20B_tokenizer.json"))

    def encode(self, src):
        return self.tok.encode(src).ids

    def decode(self, tokens):
        return self.tok.decode(tokens)


class WorldTokenizer(BaseTokenizer):

    def __init__(self):
        self.tok = RWKVTokenizer()

    def encode(self, src):
        return self.tok.encode(src)

    def decode(self, tokens):
        return self.tok.decode(tokens)
