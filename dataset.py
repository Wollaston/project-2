"""
This module implements class(es) and function(s) for dataset representation
"""
from typing import Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tokenizer import Pair, Tokenizer


@dataclass
class Sample:
    enc_inp_ids: torch.LongTensor
    dec_inp_ids: torch.LongTensor
    label_ids: torch.LongTensor


class SeqPairDataset(Dataset):
    def __init__(
            self,
            data_file: str,
            tokenizer: Tokenizer,
            max_src_len: int,
            max_tgt_len: int
    ):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        pass

