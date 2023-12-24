from transformers import LogitsProcessor
from scipy import sparse
import torch
import numpy as np


class ParaphraserLogitsProcessor(LogitsProcessor):
    """
    Avoid n_grams that is used in the original text.
    """

    def __init__(self, original_token_ids, delta, vocab_size, eos_token_id, pad_token_id, horizon=1):
        original_token_ids = np.array(original_token_ids.to("cpu"))
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
        self.delta = delta
        self.vocab_size = vocab_size
        self.bs = bs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        assert len(input_ids.shape) == 2
        bs, length = input_ids.shape
        assert bs % self.bs == 0
        bs_repeat = bs // self.bs
        horizon = self.horizon
        if length < self.horizon:
            horizon = length
        scores_shape = scores.shape
        scores = scores.view(bs, -1)
        repeated = np.zeros((bs, self.vocab_size))
        for b in range(bs):
            record_bs = b // bs_repeat
            for h in range(horizon):
                repeated[b] += self.avoid_ngrams[record_bs][h][input_ids[b, -
                                                                         h - 1].cpu()].toarray().squeeze()

        deduction = (repeated != 0) * self.delta
        scores -= torch.tensor(deduction).to(scores.device)
        return scores.view(scores_shape)


if __name__ == "__main__":
    # # test
    # orig_token=torch.tensor([[1,2,3,4,5],[4,5,3,1,1]])
    # prev_token=torch.tensor([[4,3,2,5],[4,2,5,1],[4,3,2,5],[4,2,5,1]])
    # horizon=4
    # vocab_size=6
    # delta=1
    # processor=ParaphraserLogitsProcessor(orig_token,delta,vocab_size,horizon)
    # res=processor(prev_token,torch.zeros(4,6))
    # print(res)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        "humarin/chatgpt_paraphraser_on_T5_base")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "humarin/chatgpt_paraphraser_on_T5_base").to(device)

    from utils.length_processor import ExponentialDecayLengthPenalty

    attack_prompt = "Paraphrase every word: \n"
    prompt_id = tokenizer([attack_prompt], return_tensors="pt")
    prompt_token_length = prompt_id["input_ids"].shape[-1]

    text = [
        "�s pro day. That’s where he’ll show off what he can do athletically, and also showcase his speed and quickness. “I don’t think I’ve ever really gotten the full credit I deserve from my coaches for everything I’ve done over the years,” Butler said. Butler was considered the best player at the combine by many. The 6-1, 200-pound receiver has the ability to make big plays — he averaged 17.3 yards per catch last season — and has great hands. Butler doesn’t have elite speed like some other players drafted early this spring — Alabama’s DeVonta Smith being the biggest example — but he does have impressive vertical speed, and is known for having incredible hands. “If you watch film, you see why people say he’s so talented and special,” David Montgomery told reporters Thursday afternoon. Mont",
        "though. The 8i Portal is now available on Kickstarter for $1,000. It includes everything required to get started: an 8i camera and headset; a pair of 8i glasses (sold separately); a PC capable of running the 8i software and the Oculus Rift headset ($399). You don’t need any previous knowledge about 3D modeling or programming to make your own videos using 8i’s tech. You just download the free SDK from 8i’s website, install it, then click Play. If you want to learn more about 8i’s product before deciding whether or not you want to pledge money for it, check out the 8i Portal’s page here. Business Insider Emails & Alerts Site highlights each day to your inbox. Email Address Join Follow Business Insider Australia on Facebook, Twitter, LinkedIn, and Instagram.",
        "Linux are now compatible with the latest version of X11. The list of changes for the video drivers includes the ability to play back the latest video formats supported by the various multimedia playback devices such as Blu-ray players, media centers, TVs and set-top boxes. A number of kernel patches address the issue of the kernel crashing after running out of memory if the system has a large amount of disk space reserved or the system has a lot of files mounted. There are several fixes for security vulnerabilities in various components in the Linux kernel. Two vulnerabilities affecting the kernel's file handling mechanism are fixed: CVE-2012-0178 affects the file handling mechanism, and CVE-2013-0846 concerns the kernel's file allocation mechanism. Another vulnerability affecting the kernel's file allocation mechanism, CVE-2013-0847, has already been patched. A third vulnerability impacting the file allocation mechanism, CVE-2013-0848, has already been addressed. A fourth vulnerability impacting the file allocation mechanism,",
    ]

    input_texts = [f"{attack_prompt}{t}</s>" for t in text]

    batch_inputs = tokenizer(input_texts, return_tensors="pt",
                             padding=True, truncation=True).input_ids.cuda()
    length_logits_processor = ExponentialDecayLengthPenalty(
        1.3, tokenizer.eos_token_id, batch_inputs, tokenizer.pad_token_id, prompt_token_length
    )
    paraphrase_logits_processor = ParaphraserLogitsProcessor(
        batch_inputs, 2, model.config.vocab_size, 5)

    with torch.inference_mode():
        outputs = model.generate(
            batch_inputs,
            num_beams=8,
            num_beam_groups=2,
            num_return_sequences=1,
            diversity_penalty=3.0,
            max_length=1024,
            logits_processor=[paraphrase_logits_processor],
            repetition_penalty=1.5,
            length_penalty=4,
        )
        outputs_length = torch.sum(outputs != tokenizer.pad_token_id, dim=1)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [o.strip() for o in outputs]

        outputs_baseline = model.generate(
            batch_inputs,
            num_beams=8,
            num_beam_groups=2,
            num_return_sequences=1,
            diversity_penalty=3.0,
            max_length=1024,
            logits_processor=[],
            repetition_penalty=1.5,
            length_penalty=4,
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
