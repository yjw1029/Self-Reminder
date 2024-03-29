import argparse
import jsonlines
import re
import json
from pathlib import Path

import evaluate

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

    
if __name__ == "__main__":
    # load responses of LLMs for WMT-16 dataset, compute metrics
    args = parse_args()

    out_file = Path(args.output_path)
    out_file.parent.mkdir(exist_ok=True, parents=True)

    metric = evaluate.load("sacrebleu")

    preds = []
    labels = []

    with jsonlines.open(args.result_path, 'r') as reader:
        for obj in reader:
            label = obj["label"]
            pred = obj["rslt"]

            # pred = parse_quotes(pred)

            preds.append(pred)
            labels.append(label)

    # turn off use_aggregator to get diterministic results. refer to https://github.com/huggingface/evaluate/issues/186
    eval_metric = metric.compute(predictions=preds, references=labels)
    
    print(len(labels), eval_metric)

    with open(args.output_path, 'w') as f:
        json.dump(eval_metric, f)