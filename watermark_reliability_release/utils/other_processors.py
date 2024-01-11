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

class LineBreakLogitsProcessor(LogitsProcessor):
    def __init__(self,tokenizer,min_len):
        import json
        self.eos=tokenizer.eos_token_id
        self.lb=tokenizer.encode("\n",add_special_tokens=False)
        # should be [29871, 13], but 29871 is SPIECE_UNDERLINE
        # https://github.com/huggingface/transformers/issues/26273
        self.lb=self.lb[-1]
        self.min_len=min_len
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1]<self.min_len+2:
            return scores
        consecutive_lb=(input_ids[:,-1]==self.lb)&(input_ids[:,-2]==self.lb)
        scores[consecutive_lb,:]=float("-inf")
        scores[consecutive_lb,self.eos]=1
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
