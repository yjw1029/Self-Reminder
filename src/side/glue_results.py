#%%
import os
import re
import json
import random
import evaluate
from datasets import load_dataset

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


#%%
for ti in range(5):
    for t in ["raw", "defense0"]:
        # for t in ["raw"]:
        print(f"====================== {t} =========================")
        for task_name in ["wnli", "cola", "mrpc", "sst2", "stsb", "mnli", "qqp", "qnli"]:
            # for task_name in ["qqp"]:
            metric = evaluate.load("glue", task_name)
            preds = []
            labels = []
            if not os.path.exists(f"../glue_{ti}/{t}_{task_name}.jsonl"):
                continue
            # print(f"---------------------- {task_name} --------------------")
            with open(f"../glue_{ti}/{t}_{task_name}.jsonl", "r") as f:
                for line in f:
                    rslt = json.loads(line.strip())

                    label = rslt["label"]
                    pred_str = rslt["rslt"]
                    pred = eval(f"{task_name}_get_pred")(pred_str)

                    # if pred != label:
                    #     print(label, pred_str)
                    preds.append(pred)
                    labels.append(label)

            metric.add_batch(
                predictions=preds,
                references=labels,
            )
            eval_metric = metric.compute()
            print(task_name, len(labels), eval_metric)

#%%
