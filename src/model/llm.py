from typing import Any, Callable, Tuple, List
import sys
import gc

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel

from .base import BaseModel

__all__ = [
    "LLMModel"
]


class LLMModel(BaseModel):
    def __init__(self, *, config: str|dict = None, **kwargs):
        self.config = self.load_config(config)

        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

        self.generation_config = self.load_generation_config()

    def apply_lora(self):
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config["lora_weights"],
            torch_dtype=torch.float16,
        )

    @torch.no_grad()
    def apply_delta(self):
        # load delta to cpu memory to avoid unecessary cuda memory usage
        delta = AutoModelForCausalLM.from_pretrained(
            self.config["delta_weights"],
            load_in_8bit=self.config["load_8bit"],
            torch_dtype=torch.float16,
            device_map={"": torch.device("cpu")},
            low_cpu_mem_usage=True,
        )

        for name, param in self.model.state_dict().items():
            assert name in delta.state_dict(), f"Weight {name} not in model parameters."
            param.data += delta.state_dict()[name].to(param.data.device)

        # need gc.collect() (issue https://github.com/huggingface/transformers/issues/22801)
        del delta
        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if "lora_weights" in self.config:
            self.apply_lora()

        if "delta_weights" in self.config:
            self.apply_delta()

        if not self.config["load_8bit"]:
            self.model.half()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        return self.model

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=False
        )
        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_stopping_criteria(self, input_ids):
        return None

    @torch.no_grad()
    def generate(self, data, **kwargs):
        input_ids = torch.as_tensor(data["input_ids"]).cuda()

        stopping_criteria = self.load_stopping_criteria(input_ids)

        output_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            stopping_criteria=stopping_criteria,
        )

        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.post_process(responses)
        return responses

    def post_process(self, responses: List[str]):
        return [i.strip() for i in responses]
