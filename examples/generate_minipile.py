import tyro
import datasets
from dataclasses import dataclass

from pyrwkv_tokenizer import RWKVTokenizer

from tqdm import tqdm
import numpy as np

from datasets import load_dataset

@dataclass
class Args:
    train_output_path: str = "minipile_train.npy"
    valid_output_path: str = "minipile_valid.npy"
    test_output_path: str = "minipile_test.npy"

def main():
    args = tyro.cli(Args)

    tok = RWKVTokenizer()
    
    print("getting dataset")
    ds = load_dataset("JeanKaddour/minipile")

    file_map = {
        'train': args.train_output_path,
        'validation': args.valid_output_path,
        'test': args.test_output_path,
    }

    for k in ds:
        print("loading", k)
        arrays = []
        for f in tqdm(ds[k]['text']):
            arrays.extend([[0], tok.encode(f)])
        print("generating numpy array")
        out_array = np.concatenate(arrays, dtype=np.int32)
        print("shape is", out_array.shape)
        print("saving numpy array to", file_map[k])
        np.save(file_map[k], out_array)

    
    
if __name__ == "__main__":
    main()
