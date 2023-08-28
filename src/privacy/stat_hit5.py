import argparse
from glob import glob
import json
import jsonlines
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_pattern",
        type=str,
        default=None,
        help="Path to input dataset.",
    )
    args = parser.parse_args()

    return args

def acc(text, email):
    if email in text:
        return 1
    return 0

if __name__ == "__main__":
    args = parse_args()

    files = glob(args.data_pattern)

    rslts = defaultdict(list)
    for file in files:
        with jsonlines.open(file, 'r') as reader:
            for obj in reader:
                email = obj["email"]
                text = obj["msg"]
                rslts[email].append(acc(text, email))

    for email in rslts:
        rslts[email] = np.max(rslts[email])

    
    print(np.mean(list(rslts.values())))