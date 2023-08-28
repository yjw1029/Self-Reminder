from typing import List, Dict
from collections import defaultdict

from transformers import PreTrainedTokenizer, BatchEncoding


class DefaultDataCollator:
    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)

        for example in batch_examples:
            for key in example:
                batch_rslt[key].append(example[key])

        return batch_rslt


class DataCollatorWithPadding:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding="longest",
        max_length=None,
        pad_to_multiple_of=None,
        return_attention_mask=True,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.return_tensors = return_tensors

    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)
        for example in batch_examples:
            for key in example:
                if key not in self.tokenizer.model_input_names:
                    batch_rslt[key].append(example[key])

        features = []
        for example in batch_examples:
            features.append(
                BatchEncoding({k: example[k] for k in self.tokenizer.model_input_names})
            )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors=self.return_tensors,
            verbose=True,
        )

        batch_rslt.update(features)
        return batch_rslt
