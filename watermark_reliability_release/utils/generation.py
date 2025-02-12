# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch,os
from functools import partial

# HF classes

import sys
from datasets import load_dataset, IterableDataset
from transformers import LogitsProcessorList
if __name__ == "__main__":
    sys.path.append("../")
    from data.lfqa import load_lfqa
    from data.essays import load_essays
    from data.wikitext import load_wikitext
else:
    from .data.lfqa import load_lfqa
    from .data.essays import load_essays
    from .data.wikitext import load_wikitext
from watermark_processor import WatermarkLogitsProcessor

from torch import Tensor
from tokenizers import Tokenizer

from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorWithPadding,
    LlamaForCausalLM,
    PreTrainedModel,
    PretrainedConfig
)


from utils.io import dump_json_check_path

MAX_GENERATIONS = int(10000)  # Hardcoded max length to avoid infinite loop

class Prompting:
    def __init__(self,tokenizer,prompt):
        self.tokenizer=tokenizer
        self.prompt=prompt
    def __call__(self,input_texts):
        prompts=[self.prompt+[{
            "role": "user",
            "content": it
        }] for it in input_texts]
        return [self.tokenizer.apply_chat_template(p,tokenize=False) for p in prompts]

def load_model(args, attack_model=False,device=None,config_only=False):
    """Load and return the model and tokenizer"""
    if config_only:
        CausalModel=AutoConfig
        LlamaModel=AutoConfig
        Seq2SeqModel=AutoConfig
    else:
        CausalModel=AutoModelForCausalLM
        LlamaModel=LlamaForCausalLM
        Seq2SeqModel=AutoModelForSeq2SeqLM
    if not attack_model:
        model_name_or_path = args.model_name_or_path
    else:
        model_name_or_path = args.attack_model_name
    
    cache_path_4b = "model_cache_4b/"+model_name_or_path

    args.is_seq2seq_model = any(
        [(model_type in model_name_or_path.lower())
         for model_type in ["t5", "t0"]]
    )
    args.is_decoder_only_model = any(
        [(model_type in model_name_or_path.lower())
         for model_type in ["gpt", "opt", "bloom", "llama","mistral","dolly"]]
    )
    if args.is_seq2seq_model:
        model = Seq2SeqModel.from_pretrained(model_name_or_path)
    elif args.is_decoder_only_model and not "llama" in model_name_or_path.lower():
        
        if "7b" in model_name_or_path.lower():
            if not os.path.exists(cache_path_4b):
                model = CausalModel.from_pretrained(
                    model_name_or_path, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,device_map=device if device else "auto"
                )
                os.makedirs(os.path.dirname(cache_path_4b),exist_ok=True)
                model.save_pretrained(cache_path_4b)
            else:
                model = CausalModel.from_pretrained(
                    cache_path_4b, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, device_map=device if device else "auto")
            
        elif args.load_fp16:
            model = CausalModel.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = CausalModel.from_pretrained(model_name_or_path)
    elif "llama" in model_name_or_path.lower():
        if not os.path.exists(cache_path_4b):
            model = LlamaModel.from_pretrained(
                model_name_or_path, load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,device_map=device if device else "auto"
            )
            
            os.makedirs(os.path.dirname(cache_path_4b),exist_ok=True)
            model.save_pretrained(cache_path_4b)
        else:
            model = LlamaModel.from_pretrained(
                cache_path_4b, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, device_map=device if device else "auto"
            )
        # unset temperature and top_p

        
    else:
        raise ValueError(f"Unknown model type: {model_name_or_path}")
    if device is None:
        if args.use_gpu:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if args.load_fp16 or "llama" in model_name_or_path.lower():
                pass
            else:
                model = model.to(device)
        else:
            device = "cpu"
    if not config_only:
        model.eval()

    if args.is_decoder_only_model:
        padding_side = "left"
    else:
        padding_side = None
    if "llama" in model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, add_eos_token=True)
    else:
        if padding_side:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, padding_side=padding_side
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # print(tokenizer.pad_token_id)
    # print(tokenizer.pad_token)
    if config_only:
        model_config=model
    else:
        model_config=model.config
    if hasattr(model_config, "max_position_embeddings"):
        args.model_max_length = model_config.max_position_embeddings
    else:
        args.model_max_length = 1024
    return model, tokenizer, device


def add_idx(example, idx):
    example.update({"idx": idx})
    return example


def load_hf_dataset(args):
    dataset_name, dataset_config_name = args.dataset_name, args.dataset_config_name

    if dataset_name == "lfqa":
        dataset = load_lfqa(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prefix",
                "ref_output_col_name": "gold_completion",
            }
        )
        # other args set within the load_lfqa function
    elif dataset_name == "wikitext":
        dataset = load_wikitext(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
        # other args set within the load_wikitext function
    elif dataset_name == "essays":
        dataset = load_essays(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "instructions",
                "ref_output_col_name": "essays",
            }
        )
    elif dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset(
            "./data/cml_pile.py",
            subsets=subsets,
            streaming=args.stream_dataset,
            split=None,
            ignore_verifications=True,
        )[args.dataset_split]
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=args.dataset_split,
            streaming=args.stream_dataset,
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )

    # add index to each row of dataset
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)

    # shuffle the first shuffle_buffer_size rows of streaming dataset, or whole dataset if not streaming
    # and take/select only the first n rows of the dataset (which caps the total number of pipeline iters possible)
    if isinstance(indexed_dataset, IterableDataset):
        shuffled_dataset = (
            indexed_dataset.shuffle(
                seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.take(args.limit_indices)
            if args.limit_indices is not None
            else shuffled_dataset
        )
    else:
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.select(range(args.limit_indices))
            if args.limit_indices is not None
            else shuffled_dataset
        )

    if args.limit_indices is None:
        try:
            args.limit_indices = len(limited_dataset)
        except Exception as e:
            # can't infer length of dataset, probably because it's an IterableDataset
            pass
    return limited_dataset


def check_input_lengths(
    example,
    min_sample_len=0,
    min_prompt_len=0,
    min_completion_len=0,
    max_input_len=None,
    max_new_tokens=None,
):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["baseline_completion_length"]

    if max_input_len is not None:
        assert (
            max_new_tokens is not None
        ), "need to specify max_new_tokens if max_input_length is specified"

    conds = all(
        [
            orig_sample_length >= min_sample_len,
            prompt_length >= min_prompt_len,
            real_completion_length >= min_completion_len,
            (
                ((prompt_length + max_new_tokens) <= max_input_len)
                if max_input_len is not None
                else True
            ),
        ]
    )
    return conds


def check_output_lengths(example, min_output_len=0):
    # FIXME, maybe should check baseline completion length too
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            no_wm_output_len >= min_output_len,
            w_wm_output_len >= min_output_len,
        ]
    )
    return conds

from nltk.tokenize import sent_tokenize

def find_sentence_endings(paragraphs: list[str], tokenizer, target_length, count_from_left=True,flipped=False) -> tuple[list[int]]:
    """
    Make the truncated input as short as possible.
    Count_from_left=True: keep at least target_length tokens from the left
    Count_from_left=False: remove at least target_length tokens from the right
    flipped=True: reverse the truncation and result. This is useful for the oracle.
    """

    # Assuming you have a function that finds sentence boundaries, like one from NLTK:
    # Tokenize the text into sentences to find end positions
    result = []
    formatted_inputs=[]
    for paragraph in paragraphs:
        
        keep_paragraph = ""
        if not count_from_left:
            whole_length = len(tokenizer.encode(
                paragraph, add_special_tokens=False))
            target_length = whole_length-target_length
        sentences = sent_tokenize(paragraph)
        formatted=" ".join(sentences)
        formatted_inputs.append(formatted)
        s = 0
        while len(sentences) > 0:
            sentence = sentences.pop(0)
            s += len(tokenizer.encode(sentence, add_special_tokens=False))
            keep_paragraph += sentence+" "
            if s >= target_length:
                if not flipped:
                    break
                else:
                    keep_paragraph = " ".join(sentences)+" "
        
        result.append(keep_paragraph)
    return result,formatted_inputs


def tokenize_and_truncate(
    example: dict,
    input_col_name: str = "text",
    completion_length: int = None,
    prompt_length: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    truncate_left=False,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    # tokenize
    original_input = example[input_col_name]
    if isinstance(original_input, str):
        original_input = [original_input]

    if truncate_left:
        # truncate left
        inputs_ids = inputs_ids[:, -model_max_length:]
        diff=example
        if example["untruncated_text"].shape != inputs_ids.shape:
            print(
                "Input too long for model! ",
                "Left truncating under assumption that this is the prompt+output ",
                "to be fed to the *oracle* model",
            )
        example.update({"untruncated_text": inputs_ids})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        truncated_input,formatted = find_sentence_endings(
            original_input, tokenizer, completion_length, count_from_left=False)

    elif (prompt_length is not None) and (completion_length is None):
        truncated_input,formatted = find_sentence_endings(
            original_input, tokenizer, prompt_length, count_from_left=True)
    else:
        raise ValueError(
            (
                f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                f" but got completion_length:{completion_length},prompt_length:{prompt_length}",
            )
        )
    assert len(truncated_input) == len(formatted)==1
    example.update({"untruncated_text":formatted[0]})
    example.update({"truncated_text": truncated_input[0]})
    # # truncate
    inputs_ids = tokenizer(truncated_input, return_tensors="pt")["input_ids"]
    # # logic depending on special tokens for the model
    # if "t5" in hf_model_name or "T0" in hf_model_name:
    #     inputs_ids[0, -1] = 1
    # # else: pass
    example.update({"input_ids": inputs_ids})
    return example

def tokenize_only(
    example: dict,
    input_col_name: str = "text",
    ref_output_col_name: str = None,
    tokenize_ref_output: bool = False,
    hf_model_name: str = None,
    tokenizer=None,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model
    (but don't truncate) where the dataset optionally has a secondary column
    that is the reference output to be scored against"""

    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    if ref_output_col_name is not None:
        assert ref_output_col_name in example, f"expects {ref_output_col_name} field to be present"

    # tokenize input
    input_ids = tokenizer(
        example[input_col_name], return_tensors="pt", truncation=True, max_length=model_max_length
    )["input_ids"]

    example.update({"input_ids": input_ids})

    if tokenize_ref_output:
        # NOTE not sure this logic is useful/required
        if ref_output_col_name is not None:
            # tokenize ref output
            ref_output_ids = tokenizer(
                example[ref_output_col_name],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )["input_ids"]

        tokd_input_len, tokd_ref_output_length = input_ids.shape[1], ref_output_ids.shape[1]
        if tokd_input_len + tokd_ref_output_length > model_max_length:
            # truncate the ref output
            original_ref_output_len = tokd_ref_output_length
            ref_output_ids = ref_output_ids[:,
                                            : model_max_length - tokd_input_len]
            if original_ref_output_len != ref_output_ids.shape[1]:
                print(
                    "Right truncating output, input+ref output too long for model. "
                    "Note, since this is generation time truncating the reference doesn't affect anything really."
                )
        example.update({"ref_output_ids": ref_output_ids})

    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        raise NotImplementedError("T5 style model not yet supported")

    return example


def tokenize_for_generation(
    example: dict,
    max_new_tokens: int = None,
    min_prompt_tokens: int = None,
    hf_model_name: str = None,
    tokenizer: Tokenizer = None,
    args: dict = None,
):
    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    if not args.truncate_input_for_prompt:
        tokenize_ref_output = True  # NOTE, note really sure how necessary this is
        # preprocess for model generation/completion
        example = tokenize_only(
            example,
            input_col_name=args.input_col_name,
            ref_output_col_name=args.ref_output_col_name,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            tokenize_ref_output=tokenize_ref_output,
        )
        # Parse the results of tokenization. Simple, since
        # the prompt and baseline completion are from the raw text
        re_decoded_input = example[args.input_col_name]
        decoded_baseline_completion = example[args.ref_output_col_name]
        prompt_len = example["input_ids"].shape[1]
        baseline_completion_len = example["ref_output_ids"].shape[1]
        full_sample_len = prompt_len + baseline_completion_len
        # for now, remove this here, since it's not used downstream
        example.pop("ref_output_ids")
    else:
        # preprocess for model generation/completion
        example = tokenize_and_truncate(
            example,
            completion_length=max_new_tokens,
            prompt_length=min_prompt_tokens,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
        )
        # Logic to parse the results of tokenzation and splitting to
        # construct string versions of the prompt and baseline completion
        inputs = example["input_ids"]
        prompt_len = inputs.shape[1]
        # for isolating the "gold" baseline completion
        untruncated_inputs = example.pop("untruncated_text")
        full_sample_len = len(tokenizer(untruncated_inputs)["input_ids"])

        truncated_text_len=len(example["truncated_text"])
        baseline_completion = untruncated_inputs[truncated_text_len:]
        
        baseline_completion_len = full_sample_len - prompt_len

    example.update(
        {
            "baseline_completion": baseline_completion,
            "orig_sample_length": full_sample_len,
            "prompt_length": prompt_len,
            "baseline_completion_length": baseline_completion_len,
        }
    )
    return example


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None,id_only=True):
    """collate batch of input_ids into a padded batch of tensors"""
    assert (
        input_ids[0].shape[0] == 1 and input_ids[0].shape[1] > 0
    ), "expecting batch dimension of each tensor to be 1"
    # remove batch dimension for each tensor
    input_ids = [x.squeeze(0) for x in input_ids]
    if id_only:
        return collator({"input_ids": input_ids})["input_ids"]
    else:
        return collator({"input_ids": input_ids})


def generate(
    examples,
    data_collator=None,
    generate_without_watermark=None,
    generate_with_watermark=None,
    watermark_processor=None,
    tokenizer=None,
    device=None,
    args=None,
    prompting=None,
):
    # exit()
    trunc_input=examples["truncated_text"]
    add_special_tokens=True
    if prompting:
        trunc_input=prompting(trunc_input)
        add_special_tokens=False
    if data_collator is not None:
        batch_input = tokenizer(trunc_input, return_tensors="pt",
                                padding=True,add_special_tokens=add_special_tokens)
        input_ids=batch_input["input_ids"].to(device)
        attention_mask=batch_input["attention_mask"].to(device)
    else:
        batch_input = examples["input_ids"].to(device)
        attention_mask = None
    with torch.no_grad():
        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_without_watermark = generate_without_watermark(
            input_ids=input_ids,attention_mask=attention_mask)

        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_with_watermark = generate_with_watermark(input_ids=input_ids,attention_mask=attention_mask)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,
                                                            input_ids.shape[-1]:]
        output_with_watermark = output_with_watermark[:, input_ids.shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )
    decoded_output_with_watermark = tokenizer.batch_decode(
        output_with_watermark, skip_special_tokens=True
    )
    examples.update(
        {
            "no_wm_output": decoded_output_without_watermark,
            "w_wm_output": decoded_output_with_watermark,
            "no_wm_output_length": (output_without_watermark != tokenizer.pad_token_id)
            .sum(dim=-1)
            .tolist(),
            "w_wm_output_length": (output_with_watermark != tokenizer.pad_token_id)
            .sum(dim=-1)
            .tolist(),
        }
    )

    if watermark_processor.spike_entropies is not None:
        examples["spike_entropies"] = watermark_processor._get_and_clear_stored_spike_ents()
        examples["spike_entropies"] = [
            ents[:num_toks]
            for ents, num_toks in zip(examples["spike_entropies"], examples["w_wm_output_length"])
        ]

    return examples


def get_generate_watermark(tokenizer, model, args):
    ###########################################################################
    # Construct the watermark processor
    ###########################################################################

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        store_spike_ents=args.store_spike_ents,
        select_green_tokens=True,
    )

    ###########################################################################
    # Configure the generation partials
    ###########################################################################

    gen_kwargs = dict(exponential_decay_length_penalty=(int(args.max_new_tokens), 1.05),max_new_tokens=args.max_new_tokens*1.2)
    # gen_kwargs.update(dict(exponential_decay_length_penalty=(100,1.01)))
        

    # FIXME can add typica
    if args.use_sampling:
        gen_kwargs.update(
            dict(
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                typical_p=args.typical_p,
                temperature=args.sampling_temp,
                repetition_penalty=args.repetition_penalty,
            )
        )
    else:
        gen_kwargs.update(dict(num_beams=args.num_beams,
                               repetition_penalty=args.repetition_penalty,))

    generate_without_watermark = partial(model.generate, **gen_kwargs)
    generate_with_watermark = partial(model.generate, logits_processor=LogitsProcessorList([
                                      watermark_processor]), **gen_kwargs)
    return generate_without_watermark, generate_with_watermark, watermark_processor


if __name__ == "__main__":
    print("Custom Input")
    import argparse
    from utils.submitit import str2bool

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Run watermarked huggingface LM generation ")
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="facebook/opt-2.7b",
            # default="google/flan-t5-large",
            help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--load_fp16",
            type=str2bool,
            default=True,
            help="Whether to run model in float16 precsion.",
        )
        parser.add_argument(
            "--use_gpu",
            type=str2bool,
            default=True,
            help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
        )
        parser.add_argument(
            "--use_sampling",
            type=str2bool,
            default=False,
            help=("Whether to perform sampling during generation. (non-greedy decoding)"),
        )
        parser.add_argument(
            "--sampling_temp",
            type=float,
            default=0.7,
            help="The temperature to use when generating using multinom sampling",
        )
        parser.add_argument(
            "--top_k",
            type=int,
            default=0,
            help="The top k to use when generating using top_k version of multinom sampling",
        )
        parser.add_argument(
            "--top_p",
            type=float,
            default=1.0,
            help="The top p to use when generating using top_p version of sampling",
        )
        parser.add_argument(
            "--typical_p",
            type=float,
            default=1.0,
            help="The typical p to use when generating using typical decoding version of multinom sampling",
        )
        parser.add_argument(
            "--num_beams",
            type=int,
            default=1,
            help="The number of beams to use where '1' is no beam search.",
        )
        parser.add_argument(
            "--repetition_penalty",
            type=float,
            default=1.0,
            help="The repetition penalty for transformer generation.",
        )
        parser.add_argument(
            "--generation_seed",
            type=int,
            default=None,
            help="Seed for setting the torch rng prior to generation using any decoding scheme with randomness.",
        )
        parser.add_argument(
            "--generation_batch_size",
            type=int,
            default=4,
            help="The batch size to use for generation.",
        )
        parser.add_argument(
            "--seeding_scheme",
            type=str,
            default="simple_1",
            help="The seeding procedure to use for the watermark.",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=0.25,
            help="The ratio of tokens to put in the greenlist when splitting the vocabulary",
        )
        parser.add_argument(
            "--delta",
            type=float,
            default=2.0,
            help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary at every step",
        )
        parser.add_argument(
            "--store_spike_ents",
            type=str2bool,
            default=True,
            help=(
                "Whether to store the spike entropies while generating with watermark processor. "),
        )
        parser.add_argument(
            "--verbose",
            type=str2bool,
            default=False,
            help="Whether to log the generations to stdout.",
        )
        parser.add_argument(
            "--run_name",
            type=str,
            default=None,
            help="The unique name for the run.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./output",
            help="The unique name for the run.",
        )
        parser.add_argument(
            "--overwrite",
            type=str2bool,
            default=False,
            help="Allow overwriting of old generation files at the same output location.",
        )
        parser.add_argument(
            "--max_new_tokens",
            type=int,
            default=200,
            help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
        )
        args = parser.parse_args()
        return args
    args = parse_args()

    prompt = ["If it feels like the future of AI is a rapidly changing landscape, that’s because the present innovations in the field of artificial intelligence are accelerating at such a blazing-fast pace that it’s tough to keep up.",
              "Ask me why I love dogs so much and you’ll get a two-word answer – unconditional love. And then, those two words will most likely be followed by a long rant about dogs. I’ve loved dogs long before I had one of my own.",
              "A market order is an instruction by an investor to a broker to buy or sell stock shares, bonds, or other assets at the best available price in the current financial market. It is meant to be executed as quickly as possible at the current asking price.",]
    # prompt=["Give a comprehensive description of the history of AI. ", "What is the most important breakthrough in natural language processing? Explain in detail.", "Tell me a story about a dog and a girl who loves it. "]
    model, tokenizer, device = load_model(args)
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True)[
        "input_ids"].to(device)
    examples = {"input_ids": input_ids, "prompt": prompt}
    print(input_ids)
    generate_without_watermark, generate_with_watermark, watermark_processor = get_generate_watermark(tokenizer, model,
                                                                                                      args)
    examples = generate(examples, None, generate_without_watermark,
                        generate_with_watermark, watermark_processor, tokenizer, device, args)
    # change dict of lists into list of dicts
    examples.pop("input_ids")
    examples = [dict(zip(examples, t)) for t in zip(*examples.values())]
    dump_json_check_path(examples, args)
