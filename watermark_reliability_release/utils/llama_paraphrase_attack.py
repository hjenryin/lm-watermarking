import argparse
import json
import nltk
import time
import os
import tqdm
from datasets import Dataset
import os


from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from functools import partial

from utils.generation import load_model
from paraphrase_processor import ParaphraserLogitsProcessor
from utils.llamaStructure import LlamaConversion

@torch.no_grad()
def llama_paraphrase(
    data,
    attack_prompt,
    start_idx=None,
    end_idx=None,
    paraphrase_file=".output/general_paraphrase_attacks.jsonl",
    args=None,
):
    
    output_file = paraphrase_file.split(".jsonl")[0] + "_pp" + ".jsonl"
    if args.attack_repetition_penalty!=0 or args.args.attack_length_penalty!=0:
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



    time1 = time.time()

    model, tokenizer, device = load_model(args, attack_model=True)
    convert_chat=LlamaConversion(tokenizer)
    print("Model loaded in ", time.time() - time1, "to", device)
    model.eval()
    
    # print(gen_table[0].keys())
    # exit()
    # # ['idx', 'truncated_input', 'baseline_completion', 'orig_sample_length', 'prompt_length', 'baseline_completion_length', 'no_wm_output', 'w_wm_output', 'no_wm_output_length', 'w_wm_output_length', 'spike_entropies', 'w_wm_output_attacked', 'w_wm_output_attacked_baseline', 'w_wm_output_attacked_length', 'w_wm_output_attacked_baseline_length']
    if args.full_length_possible:
        gen_table = data.to_list()
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
        print(len(gen_table))
        from utils.io import write_jsonlines
        write_jsonlines(gen_table, args.input_dir+"gen_table_expanded.jsonl")
            
    
    # start_idx = 800
    # end_idx = 802
    data = data.select(range(0, len(data))) if start_idx is None or end_idx is None else data.select(
        range(start_idx, end_idx))
    
    # iterate over data and tokenize each instance

    data=data.to_list()
    for dd in tqdm.tqdm(data):

        # print()
            # print(dd)
        if "w_wm_output_attacked" not in dd:
            if args.no_wm_attack:
                if isinstance(dd["no_wm_output"], str):
                    input_gen = dd["no_wm_output"].strip()
                else:
                    input_gen = dd["no_wm_output"][0].strip()
            else:
                if isinstance(dd["w_wm_output"], str):
                    input_gen = dd["w_wm_output"].strip()
                else:
                    input_gen = dd["w_wm_output"][0].strip()

            input_gen = " ".join(input_gen.split())
            original_input_ids.append(
                tokenizer.encode(input_gen, return_tensors="pt",add_special_tokens=False).input_ids.squeeze(0))
            final_input_text = convert_chat.chat_completion_from_content(attack_prompt,input_gen)
            input_texts.append(final_input_text)
        batch_inputs = torch.tensor(input_texts).cuda()
        _,input_length=batch_inputs.shape
        paraphrase_logits_processor = ParaphraserLogitsProcessor(
            torch.stack(original_input_ids),
            args.paraphrase_ngram_penalty,
            model.config.vocab_size,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            args.paraphrase_ngram_horizon,
        )
        
        

        with torch.inference_mode():
            outputs = model.generate(
                batch_inputs,
                num_beams=4,
                num_return_sequences=1,
                max_new_tokens=1000,
                logits_processor=[paraphrase_logits_processor],
            )[:,input_length:-1]
            
            outputs_length = outputs.shape[1]
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            outputs = [o.strip() for o in outputs]
            if "w_wm_output_attacked_baseline" not in data.keys():
                outputs_baseline = model.generate(
                    batch_inputs,
                    num_beams=4,
                    num_return_sequences=1,
                    max_new_tokens=1000,
                )[:,input_length:-1]
                outputs_baseline_length = outputs_baseline.shape[1]
                outputs_baseline = tokenizer.batch_decode(
                    outputs_baseline, skip_special_tokens=False)
                outputs_baseline = [o.strip() for o in outputs_baseline]
        w_wm_output_attacked.extend(outputs)
        w_wm_output_attacked_length.extend([outputs_length] * len(outputs))
        if "w_wm_output_attacked_baseline" not in data.keys():

            w_wm_output_attacked_baseline.extend(outputs_baseline)
            w_wm_output_attacked_baseline_length.extend(
                [outputs_baseline_length] * len(outputs_baseline))

        if args.verbose:
            for i in range(1):
                print(f"Input: {input_texts[i]}")
                print(f"Output: {outputs[i]}")
                if "w_wm_output_attacked_baseline" in data.keys():
                    print(
                        f"Output length: {outputs_length}, Target length: {len(original_input_ids[i])}")
                else:
                    print(f"Output baseline: {outputs_baseline[i]}")
                    print(
                        f"Output length: {outputs_length}, Output baseline length: {outputs_baseline_length}, Target length: {len(original_input_ids[i])}"
                    )
                print("=" * 50)

        dd["w_wm_output_attacked"] = w_wm_output_attacked[-1]
        yield dd
