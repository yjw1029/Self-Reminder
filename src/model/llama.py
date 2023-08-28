from typing import Any, Callable, Tuple, Dict, List
import sys

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from .llm import LLMModel

__all__ = [
    "LLAMAModel",
    "Alpaca",
    "Vicuna",
]


class LLAMAModel(LLMModel):
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=False
        )

        if self.tokenizer.pad_token is None:
            # LLAMA doesnot have pad token (https://github.com/huggingface/transformers/issues/22312)
            self.tokenizer.pad_token = "<unk>"
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class Alpaca(LLAMAModel):
    require_system_prompt = False

    def apply_delta(self):
        # aplaca released model changes pad token and its embedding
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["delta_weights"], use_fast=False
        )

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.tokenizer.padding_side = "left"
        return super().apply_delta()

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        user_prompt = prompt_construct_fn(example)
        message = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{user_prompt}\n\n"
            "### Response:\n"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example


class Vicuna(LLAMAModel):
    """The 1.1 version of Vicuna"""

    require_system_prompt = False

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        user_prompt = prompt_construct_fn(example)
        message = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {user_prompt} "
            f"ASSISTANT:"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example


class LLAMA2Chat(LLAMAModel):
    """The 1.1 version of Vicuna"""

    require_system_prompt = False
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=self.config["auth_token"]
        )

        if not self.config["load_8bit"]:
            self.model.half()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        return self.model

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=False, use_auth_token=self.config["auth_token"]
        )

        if self.tokenizer.pad_token is None:
            # LLAMA doesnot have pad token (https://github.com/huggingface/transformers/issues/22312)
            self.tokenizer.pad_token = "<unk>"
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer

    def chat_completion(
        self, dialog
    ):  
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": self.B_SYS
                    + dialog[0]["content"]
                    + self.E_SYS
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
                    f"{self.B_INST} {(prompt['content']).strip()} {self.E_INST} {(answer['content']).strip()} "
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
        dialog_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(dialog[-1]['content']).strip()} {self.E_INST}"
        )

        return dialog_tokens

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        user_prompt = prompt_construct_fn(example)
        dialogue = [
            {"role": "system", "content": "You are LLAMA 2-Chat, a large language model trained by Meta."},
            {"role": "user", "content": user_prompt}
        ]
        
        dialog_tokens = self.chat_completion(dialogue)
        example["input_ids"] = dialog_tokens
        example["attention_mask"] = [1] * len(dialog_tokens)

        return example

    @torch.no_grad()
    def generate(self, data, **kwargs):
        input_ids = torch.as_tensor(data["input_ids"]).cuda()

        stopping_criteria = self.load_stopping_criteria(input_ids)

        try:
            output_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                stopping_criteria=stopping_criteria,
            )
        except:
            generation_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            output_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )


        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.post_process(responses)
        return responses