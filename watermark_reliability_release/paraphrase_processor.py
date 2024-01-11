from transformers import LogitsProcessor
from scipy import sparse
import torch
import numpy as np


class FixSizedCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = {}
        self.keys = []

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = tuple(key.tolist())
        return self.cache[key]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = tuple(key.tolist())
        if key in self.cache:
            return
        if len(self.keys) >= self.max_size:
            self.cache.pop(self.keys[0])
            self.keys.pop(0)
            # del self.cache[self.keys[0]]
            # del self.keys[0]
        self.cache[key] = value
        self.keys.append(key)

    def __contains__(self, key):
        if not isinstance(key, tuple):
            key = tuple(key.tolist())
        return key in self.cache


class ParaphraserLogitsProcessor(LogitsProcessor):
    """
    Avoid n_grams that is used in the original text.
    """

    def __init__(self, original_token_ids, delta, vocab_size, eos_token_id, pad_token_id, horizon=1, max_avoid_history=0):
        original_token_ids = np.array(original_token_ids.to("cpu"))
        if pad_token_id is None:
            assert np.sum(original_token_ids == eos_token_id)==0
        else:
            original_token_ids[original_token_ids == eos_token_id] = pad_token_id
        assert horizon >= 1 and isinstance(horizon, int)
        if len(original_token_ids.shape) == 1:
            original_token_ids = original_token_ids.unsqueeze(0)
        assert len(original_token_ids.shape) == 2
        bs, length = original_token_ids.shape
        self.avoid_ngrams = [[sparse.lil_matrix(
            (vocab_size, vocab_size)) for _ in range(horizon)] for _ in range(bs)]

        for b in range(bs):
            for h in range(horizon):
                h += 1
                self.avoid_ngrams[b][h - 1][original_token_ids[b,
                                                               0: length - h], original_token_ids[b, h:]] = 1

        # self.avoid_ngrams=[[[lil.tocsr()] for lil in lils] for lils in self.avoid_ngrams]
        self.horizon = horizon
        self.max_avoid_history = max_avoid_history
        self.delta = delta
        self.vocab_size = vocab_size
        self.bs = bs
        self.repetitionCache = FixSizedCache(100)

    def getRepetition(self, input_ids: torch.LongTensor):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        assert len(input_ids.shape) == 2
        bs, length = input_ids.shape
        # remainder = torch.tensor()
        repeated = torch.zeros((bs, self.vocab_size))

        assert bs % self.bs == 0
        bs_repeat = bs // self.bs
        horizon = self.horizon
        if length < self.horizon:
            horizon = length
        for b, ids in enumerate(input_ids):
            if ids in self.repetitionCache:
                repeated[b] = self.repetitionCache[ids]
            else:
                record_bs = b // bs_repeat
                for h in range(horizon):
                    repeated[b] += self.avoid_ngrams[record_bs][h][input_ids[b, -
                                                                             h - 1].cpu()].toarray().squeeze()
                self.repetitionCache[ids] = repeated[b]
        # for i, idx in enumerate(id_match):
        #     self.repetitionCache[remainder[i]] = repeated[idx]
        return repeated != 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""
        repeated = self.getRepetition(input_ids)
        bs, length = input_ids.shape
        # This implementation is slow. Maybe use cache?
        max_avoid_history = self.max_avoid_history
        if input_ids.shape[-1] - 1 < max_avoid_history:
            max_avoid_history = input_ids.shape[-1] - 1
        ratio = torch.ones(bs)
        for _ in range(max_avoid_history):
            last_generate = input_ids[:, -1]
            input_ids = input_ids[:, :-1]
            prev_repeated = self.getRepetition(input_ids)
            # (bs,vocab_size), (bs,)
            ratio += prev_repeated[torch.arange(bs), last_generate]

        # print(ratio.shape, repeated.shape, input_ids.shape)
        deduction = (repeated != 0) * self.delta*ratio.unsqueeze(-1)
        scores -= deduction.to(scores.device)

        return scores


if __name__ == "__main__":
    # debug
    orig_token = torch.tensor([[1, 2, 3, 4, 5], [4, 5, 3, 1, 1]])

    horizon = 4
    vocab_size = 6
    delta = 1
    processor = ParaphraserLogitsProcessor(
        orig_token, delta, vocab_size, -1, -1, horizon)
    res = processor(torch.tensor([[0], [1]]), torch.zeros(2, 6))
    print(res)
    res = processor(torch.tensor([[0, 1], [1, 2]]), torch.zeros(2, 6))
    print(res)
    res = processor(torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.zeros(2, 6))
    print(res)
    res = processor(torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 1]]), torch.zeros(2, 6))
    print(res)
    res = processor(torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 1, 5]]), torch.zeros(2, 6))
    print(res)
    res = processor(torch.tensor(
        [[0, 1, 2, 3, 4, 1], [1, 2, 3, 1, 5, 3]]), torch.zeros(2, 6))
    print(res)

    exit()

    # realtime

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        "humarin/chatgpt_paraphraser_on_T5_base")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "humarin/chatgpt_paraphraser_on_T5_base").to(device)

    from utils.length_processor import ExponentialDecayLengthPenalty

    attack_prompt = "Paraphrase: \n"
    prompt_id = tokenizer([attack_prompt], return_tensors="pt")
    prompt_token_length = prompt_id["input_ids"].shape[-1]

    text = [
        "Eventually, my dream came true in the form of a little Goldendoodle puppy. Since she joined my family, life has been a little different. Anyone who’s ever owned a dog can relate to that. My daily routine now includes feeding and walking my dog, as well as the occasional trip to the park or game of fetch in the backyard. I wake up each morning from a loud bark and a wet nose in my face. It’s impossible for me to leave the house without at least one piece of dog hair on my clothing. My life has also changed because of everything that my dog has taught me.",
    ]

    input_texts = [f"{attack_prompt}{t}" for t in text]

    batch_inputs = tokenizer(input_texts, return_tensors="pt",
                             padding=True, truncation=True).input_ids.cuda()
    # length_logits_processor = ExponentialDecayLengthPenalty(
    #     1.3, tokenizer.eos_token_id, batch_inputs, tokenizer.pad_token_id, prompt_token_length
    # )
    paraphrase_logits_processor = ParaphraserLogitsProcessor(
        batch_inputs, 0.5, model.config.vocab_size, tokenizer.eos_token_id, tokenizer.pad_token_id, 5)

    with torch.inference_mode():
        outputs = model.generate(
            batch_inputs,
            num_beams=8,
            num_return_sequences=1,
            max_length=1024,
            logits_processor=[paraphrase_logits_processor],
        )
        outputs_length = torch.sum(outputs != tokenizer.pad_token_id, dim=1)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [o.strip() for o in outputs]

        outputs_baseline = model.generate(
            batch_inputs,
            num_beams=8,
            num_return_sequences=1,
            max_length=1024,
            logits_processor=[],
        )
        outputs_baseline_length = torch.sum(
            outputs_baseline != tokenizer.pad_token_id, dim=1)
        outputs_baseline = tokenizer.batch_decode(
            outputs_baseline, skip_special_tokens=True)
        outputs_baseline = [o.strip() for o in outputs_baseline]
    result = zip(outputs, outputs_baseline)
    for i in result:
        print(i)
    print(outputs_length, outputs_baseline_length)
