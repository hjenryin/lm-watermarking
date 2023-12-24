import argparse
import json
import nltk
import time
import os
import tqdm


from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from utils.generation import load_model
from utils.length_processor import ExponentialDecayLengthPenalty
from utils.repetition_ending_processor import RepetitionEndingLogitsProcessor
from paraphrase_processor import ParaphraserLogitsProcessor


@torch.no_grad()
def generate_paraphrase_paraphrase(
    data,
    attack_prompt,
    start_idx=None,
    end_idx=None,
    paraphrase_file=".output/general_paraphrase_attacks.jsonl",
    args=None,
):
    output_file = paraphrase_file.split(".jsonl")[0] + "_pp" + ".jsonl"

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

    time1 = time.time()

    model, tokenizer, device = load_model(args, attack_model=True)
    print("Model loaded in ", time.time() - time1, "to", device)
    # model.half()
    model.to(device)
    model.eval()
    # start_idx = 800
    # end_idx = 802
    data = data.select(range(0, len(data))) if start_idx is None or end_idx is None else data.select(
        range(start_idx, end_idx))

    # iterate over data and tokenize each instance
    w_wm_output_attacked = []
    w_wm_output_attacked_baseline = []
    w_wm_output_attacked_length = []
    w_wm_output_attacked_baseline_length = []
    dipper_inputs = []
    prompt_id = tokenizer([attack_prompt], return_tensors="pt")
    prompt_token_length = prompt_id["input_ids"].shape[-1]

    batch_size = 8
    num_batches = (len(data) - num_output_points) // batch_size
    last = len(data)-num_output_points-batch_size*num_batches

    for batch_idx in tqdm.tqdm(range(num_batches+1), total=num_batches):
        if batch_idx < num_batches:
            batch_data = data.select(range(num_output_points + batch_idx *
                                           batch_size, num_output_points + (batch_idx + 1) * batch_size))
        elif batch_idx == num_batches and last > 0:
            batch_data = data.select(range(num_output_points + batch_idx *
                                           batch_size, num_output_points + batch_idx * batch_size+last))
        else:
            break
        input_texts = []
        # print()
        for dd in batch_data:
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
                final_input_text = f"{attack_prompt}{input_gen}"
                input_texts.append(final_input_text)
        batch_inputs = tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
        length_logits_processor = ExponentialDecayLengthPenalty(
            args.length_align_strength, tokenizer.eos_token_id, batch_inputs, tokenizer.pad_token_id, prompt_token_length
        )
        target_length = length_logits_processor.batched_real_len
        paraphrase_logits_processor = ParaphraserLogitsProcessor(
            batch_inputs,
            args.paraphrase_ngram_penalty,
            model.config.vocab_size,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            args.paraphrase_ngram_horizon,
        )
        repetition_ending_processor = RepetitionEndingLogitsProcessor(
            args.attack_repetition_penalty)

        with torch.inference_mode():
            outputs = model.generate(
                batch_inputs,
                num_beams=8,
                num_beam_groups=2,
                num_return_sequences=1,
                diversity_penalty=3.0,
                max_length=256,
                logits_processor=[
                    length_logits_processor, paraphrase_logits_processor, repetition_ending_processor],
                repetition_penalty=args.attack_repetition_penalty,
                length_penalty=args.attack_length_penalty,
            )
            outputs_length = torch.sum(
                outputs != tokenizer.pad_token_id, dim=1)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs = [o.strip() for o in outputs]
            if "w_wm_output_attacked_baseline" not in data.features:
                outputs_baseline = model.generate(
                    batch_inputs,
                    num_beams=8,
                    num_beam_groups=2,
                    num_return_sequences=1,
                    diversity_penalty=3.0,
                    max_length=256,
                    logits_processor=[length_logits_processor,
                                      repetition_ending_processor],
                    repetition_penalty=args.attack_repetition_penalty,
                    length_penalty=args.attack_length_penalty,
                )

                outputs_baseline_length = torch.sum(
                    outputs_baseline != tokenizer.pad_token_id, dim=1)
                outputs_baseline = tokenizer.batch_decode(
                    outputs_baseline, skip_special_tokens=True)
                outputs_baseline = [o.strip() for o in outputs_baseline]
        w_wm_output_attacked.extend(outputs)
        w_wm_output_attacked_length.extend(outputs_length.cpu().tolist())
        if "w_wm_output_attacked_baseline" not in data.features:

            w_wm_output_attacked_baseline.extend(outputs_baseline)
            w_wm_output_attacked_baseline_length.extend(
                outputs_baseline_length.cpu().tolist())

        if args.verbose:
            print(f"Batch {batch_idx} done")
            for i in range(len(batch_data)):
                print(f"Input: {input_texts[i]}")
                print(f"Output: {outputs[i]}")
                if "w_wm_output_attacked_baseline" in data.features:
                    print(
                        f"Output length: {outputs_length[i]}, Target length: {target_length[i]}")
                else:
                    print(f"Output baseline: {outputs_baseline[i]}")
                    print(
                        f"Output length: {outputs_length[i]}, Output baseline length: {outputs_baseline_length[i]}, Target length: {target_length[i]}"
                    )
                print("=" * 50)

        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    if "w_wm_output_attacked" in data.features:
        data = data.remove_columns(
            ["w_wm_output_attacked", "w_wm_output_attacked_length"])
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)
    # data = data.add_column(f"dipper_inputs", dipper_inputs)
    data = data.add_column("w_wm_output_attacked_length",
                           w_wm_output_attacked_length)
    if "w_wm_output_attacked_baseline" not in data.features:
        data = data.add_column("w_wm_output_attacked_baseline",
                               w_wm_output_attacked_baseline)
        data = data.add_column(
            "w_wm_output_attacked_baseline_length", w_wm_output_attacked_baseline_length)

    return data
