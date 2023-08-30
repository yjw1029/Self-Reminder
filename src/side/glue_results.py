import argparse
import os
import re
import numpy as np
import json
import jsonlines
import random
import evaluate
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict


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

# cola
def wnli_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 1
        else:
            pred = 0
    else:
        if "not entailment" in pred.lower():
            pred = 0
        else:
            pred = 1
    return pred


def cola_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 1
        else:
            pred = 0
    else:
        if "unacceptable" in pred.lower():
            pred = 0
        else:
            pred = 1
    return pred


def mrpc_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 1
        else:
            pred = 0
    else:
        if "not semantically equivalent" in pred.lower():
            pred = 0
        else:
            pred = 1
    return pred


def mrpc_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 1
        else:
            pred = 0
    else:
        if "not semantically equivalent" in pred.lower():
            pred = 0
        else:
            pred = 1
    return pred


def sst2_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 1
        else:
            pred = 0
    else:
        if "negative" in pred.lower():
            pred = 0
        else:
            pred = 1
    return pred


def stsb_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 0
        elif pred == "B":
            pred = 1
        elif pred == "C":
            pred = 2
        elif pred == "D":
            pred = 3
        elif pred == "E":
            pred = 4
        elif pred == "F":
            pred = 5
    else:
        if "5" in pred.lower():
            pred = 5
        elif "4" in pred.lower():
            pred = 4
        elif "3" in pred.lower():
            pred = 3
        elif "2" in pred.lower():
            pred = 2
        elif "1" in pred.lower():
            pred = 1
        else:
            pred = 0
    return pred


def mnli_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 0
        elif pred == "B":
            pred = 1
        elif pred == "C":
            pred = 2
    else:
        if "entailment" in pred.lower():
            pred = 0
        elif "neutral" in pred.lower():
            pred = 1
        elif "contradiction" in pred.lower():
            pred = 2
    return pred


def qqp_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 1
        else:
            pred = 0
    else:
        if "not semantically equivalent" in pred.lower():
            pred = 0
        else:
            pred = 1
    return pred


def qnli_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 0
        else:
            pred = 1
    else:
        if "not entailment" in pred.lower():
            pred = 1
        else:
            pred = 0
    return pred


def rte_get_pred(pred):
    if len(pred) == 1:
        if pred == "A":
            pred = 0
        else:
            pred = 1
    else:
        if "not entailment" in pred.lower():
            pred = 1
        else:
            pred = 0
    return pred

if __name__ == "__main__":
    args = parse_args()

    out_file = Path(args.output_path)
    out_file.parent.mkdir(exist_ok=True, parents=True)

    preds = defaultdict(list)
    labels = defaultdict(list)

    with jsonlines.open(args.result_path, 'r') as reader:
        for obj in reader:
            label = obj["label"]
            pred_str = obj["rslt"]
            task_name = obj["task_name"]
            pred = eval(f"{task_name}_get_pred")(pred_str)

            preds[task_name].append(pred)
            labels[task_name].append(label)

    eval_metrics = {}
    for task_name in preds:
        metric = evaluate.load("glue", task_name)
        metric.add_batch(
            predictions=preds[task_name],
            references=labels[task_name],
        )

        eval_metric = metric.compute()
        for key in list(eval_metric.keys()):
            eval_metric[f"{key}_{task_name}"] = eval_metric[key]
            del eval_metric[key]

        eval_metrics.update(eval_metric)

    average = [] 
    for key in ["matthews_correlation_cola", "accuracy_sst2", "f1_mrpc", "spearmanr_stsb", "f1_qqp", "accuracy_mnli", "accuracy_qnli", "accuracy_wnli"]:
        average.append(eval_metrics[key])
    
    average = np.mean(average)
    eval_metrics["average"] = average
    
    print(len(labels), eval_metrics)

    with open(args.output_path, 'w') as f:
        json.dump(eval_metrics, f)