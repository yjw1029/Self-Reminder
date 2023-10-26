import argparse
import jsonlines
import numpy as np
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
    # load responses for CNN/DM dataset, compute metrics
    args = parse_args()

    out_file = Path(args.output_path)
    out_file.parent.mkdir(exist_ok=True, parents=True)

    metric = evaluate.load("rouge")

    preds = []
    labels = []

    with jsonlines.open(args.result_path, 'r') as reader:
        for obj in reader:
            label = obj["label"]
            pred = obj["rslt"]

            preds.append(pred)
            labels.append(label)

    # turn off use_aggregator to get diterministic results. refer to https://github.com/huggingface/evaluate/issues/186
    eval_metric = metric.compute(predictions=preds, references=labels, use_aggregator=False)

    for key in eval_metric:
        eval_metric[key] = np.mean(eval_metric[key])
    
    print(len(labels), eval_metric)

    with open(args.output_path, 'w') as f:
        json.dump(eval_metric, f)