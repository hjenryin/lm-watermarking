import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

nltk.download("punkt")
from utils.generation import load_model
from length_processor import ExponentialDecayLengthPenalty
from ..paraphrase_processor import ParaphraserLogitsProcessor


def generate_paraphrase_paraphrase(
    data,
    attack_prompt,
    sent_interval=3,
    start_idx=None,
    end_idx=None,
    paraphrase_file=".output/general_paraphrase_attacks.jsonl",
    args=None,
):
    if sent_interval == 1:
        paraphrase_file = paraphrase_file.split(".jsonl")[0] + "_sent" + ".jsonl"

    output_file = paraphrase_file.split(".jsonl")[0] + "_pp" + ".jsonl"

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            num_output_points = len([json.loads(x) for x in f.read().strip().split("\n")])
    else:
        num_output_points = 0
    print(f"Skipping {num_output_points} points")

    time1 = time.time()

    model, tokenizer, device = load_model(args)
    print("Model loaded in ", time.time() - time1)
    # model.half()
    model.to(device)
    model.eval()

    data = data.select(range(0, len(data))) if start_idx is None or end_idx is None else data.select(range(start_idx, end_idx))

    # iterate over data and tokenize each instance
    w_wm_output_attacked = []
    w_wm_output_attacked_baseline = []
    dipper_inputs = []
    prompt_id = tokenizer([attack_prompt], return_tensors="pt")
    prompt_token_length = prompt_id["input_ids"].shape[-1]

    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        if idx < num_output_points:
            continue
        # tokenize prefix
        if "w_wm_output_attacked" not in dd:
            # paraphrase_outputs = {}

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

            # remove spurious newlines
            input_gen = " ".join(input_gen.split())
            sentences = sent_tokenize(input_gen)
            prefix = " ".join(dd["truncated_input"].replace("\n", " ").split())
            output_text = ""
            final_input_text = ""

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])

                final_input_text = f"{attack_prompt}: {curr_sent_window}"

                if idx == 0:
                    print(final_input_text)

                final_input = tokenizer([final_input_text], return_tensors="pt")
                final_input = {k: v.to(device) for k, v in final_input.items()}

                with torch.inference_mode():
                    outputs = model.generate(
                        **final_input,
                        num_beams=8,
                        num_beam_groups=2,
                        num_return_sequences=1,
                        diversity_penalty=3.0,
                        max_length=1024,
                        length_penalty=1.3,
                        logits_processor=[
                            ExponentialDecayLengthPenalty(1.3, tokenizer.eos_token_id, final_input.token_ids.shape[-1]),
                            ParaphraserLogitsProcessor(final_input.token_ids, 2, tokenizer.vocab_size, 5),
                        ],
                    )
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prefix += " " + outputs[0]
                output_text += " " + outputs[0]

            w_wm_output_attacked_baseline.append(output_text.strip())
            dipper_inputs.append(final_input_text)

        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)
    data = data.add_column(f"dipper_inputs", dipper_inputs)

    return data
