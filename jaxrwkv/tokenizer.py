from tokenizers import Tokenizer

import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

"""
RWKV tokenizer, mostly borrowed from https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py
"""
class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for line in lines:
            idx = int(line[:line.index(' ')])
            x = eval(line[line.index(' '):line.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(line[line.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except Exception as ex:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except Exception as ex:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()


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
        self.tok = RWKV_TOKENIZER(os.path.join(dir_path, "tok_files/rwkv_vocab_v20230424.txt"))

    def encode(self, src):
        return self.tok.encode(src)

    def decode(self, tokens):
        return self.tok.decode(tokens)
