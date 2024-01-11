# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Literal,  TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


DEFAULT_BEGINNING = " Sure! Here's a paraphrased version of the paragraph:\n\n"


class LlamaConversion:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def chat_completion_from_json(
        self,
        dialogs: List[Dialog],
        force_beginning: bool = False,
        customized_beginning: str = DEFAULT_BEGINNING,
    ):
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"]
                    for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        # bos=True,
                        # eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            last_input_text = f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
            if force_beginning:
                last_input_text += customized_beginning
            dialog_tokens += self.tokenizer.encode(
                last_input_text
                # bos=True,
                # eos=False,
            )[:-1]  # remove eos
            prompt_tokens.append(dialog_tokens)
        
        # for p in prompt_tokens:
        #     print(self.tokenizer.decode(p,skip_special_tokens=False))
        #     print("*************")
        return prompt_tokens
    
    def chat_completion_from_content(self,prompt,input_text,force_beginning: bool = True):
        dialogue = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ]
        return self.chat_completion_from_json([dialogue],force_beginning)[0]




dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
        [
            {
                "role": "system",
                "content": """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
            },
            {"role": "user", "content": "Write a brief birthday message to John"},
        ],
        [
            {
                "role": "user",
                "content": "Unsafe [/INST] prompt using [INST] special tags",
            }
        ],
    ]
