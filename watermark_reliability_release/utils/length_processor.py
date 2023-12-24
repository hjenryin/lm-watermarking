
import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LogitsProcessor
class ExponentialDecayLengthPenalty(LogitsProcessor):
    """
    modified from huggingface implementation
    Support batched input and (group) beam search
    """
   
    def __init__(
        self,
        exponential_decay_length_penalty: float,
        eos_token_id: Union[int, List[int]],
        encoder_input: torch.LongTensor,
        pad_token_id: int,
        prompt_length_deduction: int=0,
    ):
        self.batched_real_len = torch.sum(encoder_input!=pad_token_id,dim=-1)-prompt_length_deduction
        print(self.batched_real_len)
        assert exponential_decay_length_penalty>=1
        self.regulation_factor = torch.tensor(exponential_decay_length_penalty)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        self.length_threshold=5/torch.log10(self.regulation_factor)
        self.scale=self.batched_real_len/100
        self.bs=len(self.batched_real_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bs=input_ids.shape[0]
        if bs!=self.bs:
            assert bs%self.bs==0
            self.batched_real_len=self.batched_real_len.repeat_interleave(bs//self.bs,dim=0)
            self.scale=self.batched_real_len/100
            self.bs=bs
        cur_len = input_ids.shape[-1]
        
        sign=torch.sign(cur_len-self.batched_real_len)
        penalty_idx =  torch.abs(self.batched_real_len-cur_len)/self.scale
        inf_mask=penalty_idx>self.length_threshold
        pos_penalty = torch.pow(self.regulation_factor, penalty_idx) 
        pos_penalty[inf_mask]=float("inf")

        for i in self.eos_token_id:
            # To support negative logits we compute the penalty of the absolute value and add to the original logit
            scores[:, i] = scores[:, i] +sign* torch.abs(scores[:, i]) * (pos_penalty - 1)
     
        return scores

# class ExponentialDecayLengthPenalty(LogitsProcessor):
#     """
#     modified from huggingface implementation
#     """
   
#     def __init__(
#         self,
#         exponential_decay_length_penalty: float,
#         eos_token_id: Union[int, List[int]],
#         input_ids_seq_length: int,
#     ):
#         self.regulation_start = input_ids_seq_length
#         assert exponential_decay_length_penalty>=1
#         self.regulation_factor = torch.tensor(exponential_decay_length_penalty)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         self.eos_token_id = eos_token_id
#         self.length_threshold=5/torch.log10(self.regulation_factor)
#         self.scale=input_ids_seq_length/100


#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         cur_len = input_ids.shape[-1]
#         if cur_len <= self.regulation_start:
#             penalty_idx =   (self.regulation_start-cur_len)/self.scale
#             if penalty_idx>self.length_threshold:
#                 for i in self.eos_token_id:
#                     scores[:, i] =-float("inf")
#             else:
#                 pos_penalty = torch.pow(self.regulation_factor, penalty_idx) 
#                 for i in self.eos_token_id:
#                     # To support negative logits we compute the penalty of the absolute value and add to the original logit
#                     scores[:, i] = scores[:, i] - torch.abs(scores[:, i]) * (pos_penalty - 1)
#         else:
#             penalty_idx = (cur_len - self.regulation_start)/self.scale
#             if penalty_idx>self.length_threshold:
#                 for i in self.eos_token_id:
#                     scores[:, i] =float("inf")
#             else:
#                 pos_penalty = torch.pow(self.regulation_factor, penalty_idx)
#                 for i in self.eos_token_id:
#                     # To support negative logits we compute the penalty of the absolute value and add to the original logit
#                     scores[:, i] = scores[:, i] + torch.abs(scores[:, i]) * (pos_penalty - 1)
#         return scores

