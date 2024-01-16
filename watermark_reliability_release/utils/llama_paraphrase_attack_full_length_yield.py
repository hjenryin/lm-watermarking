import argparse
import json
import nltk
import time
import os
import tqdm
from datasets import Dataset
import os
from datasets.utils import logging


from transformers import T5Tokenizer, T5ForConditionalGeneration,BatchEncoding
import torch
from functools import partial

from utils.generation import load_model
from paraphrase_processor import ParaphraserLogitsProcessor
from utils.llamaStructure import LlamaConversion
from utils.other_processors import LineBreakLogitsProcessor

from torch.multiprocessing import  Pool

model=None
tokenizer=None
convert_chat=None
CACHE=0
CAL=1
ERROR=2

  
def load_model_mp(i,args):
    global model,tokenizer,convert_chat

    model, tokenizer, device = load_model(
        args, attack_model=True, device=f"cuda:{i}")
    convert_chat = LlamaConversion(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model.eval()
    
@torch.no_grad()
def inference(dd,args,attack_prompt,recover:Dataset):
    global model,tokenizer,convert_chat
    logging.disable_progress_bar()
    input_texts = []
    original_input_ids = []
    if "w_wm_output_attacked" not in dd:

        if isinstance(dd["w_wm_output"], str):
            input_gen = dd["w_wm_output"].strip()
        else:
            input_gen = dd["w_wm_output"][0].strip()
        if recover:
            if input_gen in recover["w_wm_output"]:
                result = recover.filter(lambda x:x["w_wm_output"]==input_gen)
                return result.to_list()[0],CACHE
        input_gen = " ".join(input_gen.split())
        original_input_ids.append(
            tokenizer.encode(input_gen, add_special_tokens=False))
        final_input_text = convert_chat.chat_completion_from_content(attack_prompt,input_gen,force_beginning=True)
        input_texts.append(final_input_text)
    batch_inputs = torch.tensor(input_texts).to(model.device)
    _,input_length=batch_inputs.shape
    lb_processor=LineBreakLogitsProcessor(tokenizer,input_length) # Avoid stuff like "I hope this helps!"
    torch.cuda.empty_cache()
    paraphrase_logits_processor = ParaphraserLogitsProcessor(
        torch.tensor(original_input_ids),
        args.paraphrase_ngram_penalty,
        model.config.vocab_size,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        args.paraphrase_ngram_horizon,
    )
    try:
        with torch.inference_mode():
            outputs = model.generate(
                batch_inputs,
                # num_beams=3,
                num_return_sequences=1,
                max_new_tokens=1000,
                logits_processor=[paraphrase_logits_processor,lb_processor],
                do_sample=True
            )[0][input_length:-1]
            
            outputs_length = outputs.shape[0]
            outputs = tokenizer.decode(outputs, skip_special_tokens=False)
            if "w_wm_output_attacked_baseline" not in dd.keys():
                outputs_baseline = model.generate(
                    batch_inputs,
                    # num_beams=3,
                    num_return_sequences=1,
                    max_new_tokens=1000,
                    logits_processor=[lb_processor],
                    do_sample=True
                )[0][input_length:-1]
                outputs_baseline_length = outputs_baseline.shape[0]
                outputs_baseline = tokenizer.decode(
                    outputs_baseline, skip_special_tokens=False)
        w_wm_output_attacked=outputs
        w_wm_output_attacked_length=outputs_length
        if "w_wm_output_attacked_baseline" not in dd.keys():
            w_wm_output_attacked_baseline=outputs_baseline
            w_wm_output_attacked_baseline_length=outputs_baseline_length

        if args.verbose:
            for i in range(1):
                print(f"Input: {input_gen}")
                print(f"Output: {outputs}")
                if "w_wm_output_attacked_baseline" in dd.keys():
                    print(
                        f"Output length: {outputs_length}, Target length: {len(original_input_ids[i])}")
                else:
                    print(f"Output baseline: {outputs_baseline}")
                    print(
                        f"Output length: {outputs_length}, Output baseline length: {outputs_baseline_length}, Target length: {len(original_input_ids[i])}"
                    )
                print("=" * 50)

        dd["w_wm_output_attacked"] = w_wm_output_attacked
        dd["w_wm_output_attacked_length"] = w_wm_output_attacked_length
        if "w_wm_output_attacked_baseline" not in dd.keys():
            dd["w_wm_output_attacked_baseline"] = w_wm_output_attacked_baseline
            dd["w_wm_output_attacked_baseline_length"] = w_wm_output_attacked_baseline_length
        return dd,CAL
    except RuntimeError as e:
        print(dd["idx"],dd["w_wm_output_length"],e)
        return dd,ERROR


@torch.no_grad()
def batched_inference(dd, args, attack_prompt, recover: Dataset):
    global model, tokenizer, convert_chat
    logging.disable_progress_bar()
    converted_input_text = []
    original_input_ids = []
    dd_list=dd
    inputs_gen=[]
    if recover:
        results=[]
        for dd in dd_list:
            if isinstance(dd["w_wm_output"], str):
                input_gen = dd["w_wm_output"].strip()
            else:
                input_gen = dd["w_wm_output"][0].strip()
            if recover:
                if input_gen in recover["w_wm_output"]:
                    result = recover.filter(
                        lambda x: x["w_wm_output"] == input_gen)
                    results.append(result.to_list()[0]) 
        if len(results)==len(dd_list):
            return results,CACHE
    for dd in dd_list:
        if isinstance(dd["w_wm_output"], str):
            input_gen = dd["w_wm_output"].strip()
        else:
            input_gen = dd["w_wm_output"][0].strip()
        input_gen = " ".join(input_gen.split())
        original_input_ids.append(
            tokenizer.encode(input_gen, add_special_tokens=False))
        final_input_id = convert_chat.chat_completion_from_content(
            attack_prompt, input_gen, force_beginning=True)
        converted_input_text.append(tokenizer.decode(final_input_id,skip_special_tokens=False))
        inputs_gen.append(input_gen)
    batch_inputs = tokenizer(
        converted_input_text, add_special_tokens=False,padding=True,return_attention_mask=True,return_tensors="pt").to(model.device)
    # FIXME inefficient

    dd=Dataset.from_list(dd_list)

    bs, input_length = batch_inputs['input_ids'].shape
    # Avoid stuff like "I hope this helps!"
    lb_processor = LineBreakLogitsProcessor(tokenizer, input_length)
    torch.cuda.empty_cache()
    paraphrase_logits_processor = ParaphraserLogitsProcessor(
        batch_inputs["input_ids"],
        args.paraphrase_ngram_penalty,
        model.config.vocab_size,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        args.paraphrase_ngram_horizon,
    )
    try:
        with torch.inference_mode():
            from transformers import generation_utils
            outputs = model.generate(
                **batch_inputs,
                # num_beams=3,
                num_return_sequences=1,
                max_new_tokens=1000,
                logits_processor=[paraphrase_logits_processor, lb_processor],
                do_sample=True
            )
            outputs=outputs[:,input_length:-1]

            outputs_length = torch.sum(outputs != tokenizer.pad_token_id, dim=-1)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if "w_wm_output_attacked_baseline" not in dd.features:
                outputs_baseline = model.generate(
                    **batch_inputs,
                    # num_beams=3,
                    num_return_sequences=1,
                    max_new_tokens=1000,
                    logits_processor=[lb_processor],
                    do_sample=True
                )[:,input_length:-1]
                outputs_baseline_length = torch.sum(
                    outputs_baseline != tokenizer.pad_token_id, dim=-1)
                outputs_baseline = tokenizer.batch_decode(
                    outputs_baseline, skip_special_tokens=True)
        w_wm_output_attacked = outputs
        w_wm_output_attacked_length = outputs_length.tolist()
        if "w_wm_output_attacked_baseline" not in dd.features:
            w_wm_output_attacked_baseline = outputs_baseline
            w_wm_output_attacked_baseline_length = outputs_baseline_length.tolist()

        if args.verbose:
            for i in range(bs):
                print(f"Input: {inputs_gen[i]}")
                print(f"Output: {outputs[i]}")
                if "w_wm_output_attacked_baseline" in dd.features:
                    print(
                        f"Output length: {outputs_length[i]}, Target length: {len(original_input_ids[i])}")
                else:
                    print(f"Output baseline: {outputs_baseline[i]}")
                    print(
                        f"Output length: {outputs_length[i]}, Output baseline length: {outputs_baseline_length[i]}, Target length: {len(original_input_ids[i])}"
                    )
                print("=" * 50)

        dd.add_column("w_wm_output_attacked" ,w_wm_output_attacked)
        dd.add_column("w_wm_output_attacked_length" ,w_wm_output_attacked_length)
        if "w_wm_output_attacked_baseline" not in dd.features:
            dd.add_column("w_wm_output_attacked_baseline" ,w_wm_output_attacked_baseline)
            dd.add_column("w_wm_output_attacked_baseline_length" ,w_wm_output_attacked_baseline_length)
        return dd.to_list(), CAL
    except RuntimeError as e:
        print(dd["idx"], dd["w_wm_output_length"], e)
        return dd.to_list(), ERROR


@torch.no_grad()
def llama_paraphrase(
    data,
    attack_prompt,
    start_idx=None,
    end_idx=None,
    paraphrase_file=".output/general_paraphrase_attacks.jsonl",
    args=None,
    recover=None
):
    
    if recover:
        recover=Dataset.from_list(recover)
    # recovered_input=recover["w_wm_output"]
    # data_input=data["w_wm_output"]
    # print(len(recovered_input),len(data_input))
    # for ri in recovered_input:
    #     print(next(i for i in range(len(data_input)) if data_input[i]==ri),end=" ")

    output_file = paraphrase_file.split(".jsonl")[0] + "_pp" + ".jsonl"
    if args.attack_repetition_penalty!=0 or args.attack_length_penalty!=0:
        print("When using a model as powerful as llama, some parameters are ignored.")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            num_output_points = len([json.loads(x)
                                    for x in f.read().strip().split("\n")])
    else:
        num_output_points = 0
    print(f"Skipping {num_output_points} points")
    if num_output_points == 0:
        if "w_wm_output_attacked" in data.features:
            data = data.remove_columns(
                ["w_wm_output_attacked", "w_wm_output_attacked_length"])
        if "w_wm_output_attacked_baseline" in data.features:
            data = data.remove_columns(
                ["w_wm_output_attacked_baseline", "w_wm_output_attacked_baseline_length"])



    
    

    # # ['idx', 'truncated_input', 'baseline_completion', 'orig_sample_length', 'prompt_length', 'baseline_completion_length', 'no_wm_output', 'w_wm_output', 'no_wm_output_length', 'w_wm_output_length', 'spike_entropies', 'w_wm_output_attacked', 'w_wm_output_attacked_baseline', 'w_wm_output_attacked_length', 'w_wm_output_attacked_baseline_length']
    if args.full_length_possible:
        model, tokenizer, device = load_model(args, attack_model=True,config_only=True)
        
        gen_table = data.to_list()
        old_length=len(gen_table)
        from nltk.tokenize import sent_tokenize
        new_gen_table = []
        for ex in gen_table:
            output_sent = ex['w_wm_output']
            output_sent_list = sent_tokenize(output_sent)
            atk_append = ""
            for atk in output_sent_list:
                new_ex = ex.copy()
                atk_append += " "+atk if atk_append != "" else atk
                new_ex['w_wm_output'] = atk_append
                tokens=tokenizer(atk_append, add_special_tokens=False)['input_ids']
                new_ex['w_wm_output_length'] = len(tokens)
                if len(new_gen_table) < 2:
                    back=tokenizer.decode(tokens,skip_special_tokens=False)
                    print(back)
                    
                new_gen_table.append(new_ex)
        gen_table = new_gen_table
        data = Dataset.from_list(gen_table)
        for i in range(5):
            print(gen_table[i]["w_wm_output"],gen_table[i]["w_wm_output_length" ])
        print("Expand from",old_length,"to",len(gen_table))
        from utils.io import write_jsonlines
        write_jsonlines(gen_table, args.input_dir+"/gen_table_expanded.jsonl")
            
    data = data.filter(lambda x: x["w_wm_output_length"] <450)
    # start_idx = 800
    # end_idx = 802
    data = data.select(range(0, len(data))) if start_idx is None or end_idx is None else data.select(
        range(start_idx, end_idx))
    
    # iterate over data and tokenize each instance

    data=data.to_list()
    device_count=torch.cuda.device_count()

    pbar=tqdm.tqdm(total=len(data))
    shortest_failure=5000
    shortest_failure_CUDA=5000
    longest_success=0
    longest_success_add=0
    with Pool(device_count) as p:
        time1=time.time()
        p.map(partial(load_model_mp,args=args),range(device_count))
        # load_model_mp(0,args)
        print("Model loaded in ", time.time() - time1)
        
        BS=1
        # data=torch.utils.data.DataLoader(data,batch_size=BS,collate_fn=lambda x:x)
        
        
        # for dd in data:
        #     result=inference(dd,args,attack_prompt)
        cached=0
        for results in p.imap(partial(inference,args=args,attack_prompt=attack_prompt,recover=recover),data):
            results,flag=results
            if not isinstance(results,list):
                results=[results]
            if flag==CACHE:
                cached+=BS
            pbar.update(BS)
            for result in results:
                if "w_wm_output_attacked" in result.keys():
                    atk,orig,base=result["w_wm_output_attacked_length"],result["w_wm_output_length"],result["w_wm_output_attacked_baseline_length"]
                    longest_success_add = max(longest_success_add, atk+orig)
                    longest_success_add = max(longest_success_add, base+orig)
                    longest_success = max(longest_success, orig)
                    if recover:
                        pbar.set_description(
                            f"Success: {longest_success_add} / {longest_success}, Failure: {shortest_failure} / {shortest_failure_CUDA}, Cache: {cached} / {len(recover)}")
                    else:
                        pbar.set_description(
                            f"Success: {longest_success_add} / {longest_success}, Failure: {shortest_failure} / {shortest_failure_CUDA}")
                    yield result
                else:
                    shortest_failure=min(shortest_failure,result["w_wm_output_length"])
                    if result["w_wm_output_length"]>longest_success:
                        if shortest_failure_CUDA<longest_success:
                            shortest_failure_CUDA=5000
                        else:
                            shortest_failure_CUDA = min(
                                shortest_failure_CUDA, result["w_wm_output_length"])
                        
                    if recover:
                        pbar.set_description(
                            f"Success: {longest_success_add} / {longest_success}, Failure: {shortest_failure} / {shortest_failure_CUDA}, Cache: {cached} / {len(recover)}")
                    else:
                        pbar.set_description(
                            f"Success: {longest_success_add} / {longest_success}, Failure: {shortest_failure} / {shortest_failure_CUDA}")
                    
                    continue

            
            