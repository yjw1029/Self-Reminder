import argparse
import jsonlines
import json
import numpy as np
from pathlib import Path

import evaluate
from rouge import RougeRecall

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="Path of attack responses for evaluation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to export evaluation results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size to inference."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from previous stored file. If the file does not exist test from scracth.",
    )

    args = parser.parse_args()

    return args

class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


if __name__ == "__main__":
    # load responses of LLMs for SQuAD dataset, compute metrics
    args = parse_args()

    out_file = Path(args.output_path)
    out_file.parent.mkdir(exist_ok=True, parents=True)

    # Use recall instead of fmeature defined in huggingface evaluate
    metric1 = RougeRecall()
    metric2 = evaluate.load("rouge")

    preds = []
    labels = []

    with jsonlines.open(args.result_path, 'r') as reader:
        for obj in reader:
            label = obj["label"]["text"]
            pred = obj["rslt"]

            if len(label) == 0:
                label = ["Unknown"]

            preds.append(pred)
            labels.append(label)

    # turn off use_aggregator to get diterministic results. refer to https://github.com/huggingface/evaluate/issues/186
    eval_metric1 = metric1.compute(predictions=preds, references=labels, use_aggregator=False)
    eval_metric2 = metric2.compute(predictions=preds, references=labels, use_aggregator=False)

    for key in list(eval_metric1.keys()):
        eval_metric1[f"{key}_recall"] = np.mean(eval_metric1[key])
        del eval_metric1[key]

    for key in list(eval_metric2.keys()):
        eval_metric2[f"{key}_f"] = np.mean(eval_metric2[key])
        del eval_metric2[key]

    eval_metric = {}
    eval_metric.update(eval_metric1)
    eval_metric.update(eval_metric2)

    print(len(labels), eval_metric)

    with open(args.output_path, 'w') as f:
        json.dump(eval_metric, f)
