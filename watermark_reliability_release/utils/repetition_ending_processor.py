import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LogitsProcessor


class RepetitionEndingLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bs = input_ids.shape[0]
        cur_len = input_ids.shape[-1]
        if cur_len == 0:
            return scores
        last_token = input_ids[:, -1]
        penalty = torch.zeros_like(scores)
        penalty.scatter_(1, last_token.unsqueeze(1), self.penalty)
        scores -= penalty
        # print(penalty)
        return scores


if __name__ == "__main__":
    # test
    prev_token = torch.tensor([[4, 3, 2, 3], [4, 2, 5, 1], [4, 3, 2, 5], [4, 2, 5, 1]])
    horizon = 4
    penalty = 1
    processor = RepetitionEndingLogitsProcessor(penalty)
    scores = torch.rand(4, 6)
    processor(prev_token, scores)
    pass
